from dataclasses import dataclass


@dataclass
class TransActConfig:
    """
    Configuration class to build a TransAct PyTorch module.

    :param seq_len: Length of the input sequence
    :param time_window_ms: Time window in milliseconds for random window mask
    :param latest_n_emb: Number of latest embeddings to use in output
    :param concat_candidate_emb: Whether to concatenate candidate embeddings with user sequence
    :param concat_max_pool: Whether to apply max pooling to the output of the transformer encoder and append it to output
    :param action_vocab: Vocabulary of user actions
    :param action_emb_dim: Dimension of user action embeddings
    :param item_emb_dim: Dimension of item embeddings
    :param num_layer: Number of TransformerEncoderLayer
    :param nhead: Number of heads in the TransformerEncoderLayer
    :param dim_feedforward: Feed forward dimension of the TransformerEncoderLayer
    """

    seq_len: int = 100
    time_window_ms: int = 1000 * 60 * 60 * 1
    latest_n_emb: int = 10
    concat_candidate_emb: bool = True
    concat_max_pool: bool = True
    action_vocab: list = range(0, 20)
    action_emb_dim: int = 32
    item_emb_dim: int = 32
    num_layer: int = 2
    nhead: int = 2
    dim_feedforward: int = 32
