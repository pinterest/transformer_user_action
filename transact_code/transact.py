import torch
import torch.nn as nn
import logging
from typing import Dict, List, Final
import random
from transact_config import TransActConfig


class TransAct(nn.Module):
    """
    TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest
    """

    def __init__(
        self,
        transact_config: TransActConfig,
    ):
        """
        Module Initialization
        """
        super().__init__()
        self.transact_config = transact_config
        self.action_emb_dim: Final[int] = self.transact_config.action_emb_dim
        self.item_emb_dim: Final[int] = self.transact_config.item_emb_dim
        self.concat_candidate_emb: Final[
            bool
        ] = self.transact_config.concat_candidate_emb
        self.time_window_ms: Final[int] = self.transact_config.time_window_ms
        self.seq_len: Final[int] = self.transact_config.seq_len
        self.latest_n_emb: Final[int] = self.transact_config.latest_n_emb
        self.action_vocab: Final[list] = self.transact_config.action_vocab
        if self.concat_candidate_emb:
            transformer_in_dim = self.action_emb_dim + self.item_emb_dim * 2
        else:
            transformer_in_dim = self.action_emb_dim + self.item_emb_dim

        self.register_buffer("action_type_lookup", self.convert_vocab_to_idx())
        self.action_emb_module = nn.Embedding(
            len(self.action_vocab), self.action_emb_dim, padding_idx=0
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_in_dim,
            nhead=self.transact_config.nhead,
            dim_feedforward=self.transact_config.dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.transact_config.num_layer
        )
        if self.transact_config.concat_max_pool:
            self.out_linear = nn.Linear(transformer_in_dim, transformer_in_dim)

    def convert_vocab_to_idx(self) -> torch.Tensor:
        logging.info(f"Action used: {self.action_vocab}")
        t = torch.zeros(100, dtype=torch.long)
        i = 0
        for id in sorted(self.action_vocab):
            t[id + 1] = i
            i += 1
        return t

    def forward(
        self,
        action_type_seq: torch.Tensor,
        item_embedding_seq: torch.Tensor,
        action_time_seq: torch.Tensor,
        request_time: torch.Tensor,
        item_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param action_type_seq: Tensor of shape (batch_size, seq_len) representing the sequence of action types
        :param item_embedding_seq: Tensor of shape (batch_size, seq_len, item_emb_dim) representing the sequence of item embeddings
        :param action_time_seq: Tensor of shape (batch_size, seq_len) representing the sequence of action times
        :param request_time: Tensor of shape (batch_size, 1) representing the request time
        :param item_embedding: Tensor of shape (batch_size, item_emb_dim) representing the current item embedding
        :return: Tensor of shape (batch_size, latest_n_emb+1) representing the output of the forward pass
        """
        # step 1: get the latest N actions from sequence features
        action_type_seq = action_type_seq[:, : self.seq_len]
        item_embedding_seq = item_embedding_seq[:, : self.seq_len, :]
        action_time_seq = action_time_seq[:, : self.seq_len]

        # step 2: get action embedding
        action_type_seq = self.action_type_lookup[action_type_seq + 1]
        action_emb_tensor = self.action_emb_module(action_type_seq)

        # step 3: create mask that tells transformer which position to be ignored by the attention.
        # mask padded positions
        key_padding_mask = action_type_seq <= 0

        # mask actions that happened in a time window before the request time
        request_time = request_time.unsqueeze(-1).expand(-1, self.seq_len)

        # randomly sample a time window to introduce randomness
        rand_time_window_ms = random.randint(0, self.time_window_ms)
        short_time_window_idx_trn = (
            request_time - action_time_seq
        ) < rand_time_window_ms
        # use all the actions during inference
        short_time_window_idx_eval = (request_time - action_time_seq) < 0

        # adjust the mask accordingly
        if self.training:
            key_padding_mask = self._adjust_mask(
                key_padding_mask, short_time_window_idx_trn
            )
        else:
            key_padding_mask = self._adjust_mask(
                key_padding_mask, short_time_window_idx_eval
            )

        # step 4: stack seq embedding with action embedding and candidate embedding
        action_pin_emb = torch.cat((action_emb_tensor, item_embedding_seq), dim=2)

        if self.concat_candidate_emb:
            # Stack the candidate pin embedding with the sequence embedding
            item_embedding_expanded = item_embedding.unsqueeze(1).expand(
                -1, self.seq_len, -1
            )
            action_pin_emb = torch.cat(
                (action_pin_emb, item_embedding_expanded), dim=-1
            )

        # step 5: pass the sequence to transformer
        tfmr_out = self.transformer_encoder(
            src=action_pin_emb, src_key_padding_mask=key_padding_mask
        )

        # step 6: process the transformer output
        output_concat = []
        if self.transact_config.concat_max_pool:
            # Apply max pooling to the transformer output
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        if self.latest_n_emb > 0:
            tfmr_out = tfmr_out[:, : self.latest_n_emb]
        output_concat.append(tfmr_out.flatten(1))
        return torch.cat(output_concat, dim=1)

    def _adjust_mask(self, mask: torch.Tensor, short_time_window_idx: torch.Tensor):
        # make sure not all actions in the sequence are masked
        mask = torch.bitwise_or(mask, short_time_window_idx)
        mask[:, 0] = torch.zeros(mask.shape[0], dtype=mask.dtype, device=mask.device)
        new_attn_mask = torch.zeros_like(mask, dtype=torch.float, device=mask.device)
        new_attn_mask.masked_fill_(mask, float("-inf"))
        return new_attn_mask
