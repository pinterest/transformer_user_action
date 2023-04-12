import torch
from transact import TransAct
from transact_config import TransActConfig


def test_run_transact():
    action_vocab = list(range(0, 20))
    full_seq_len = 100
    test_batch_size = 8
    action_emb_dim = 32
    item_emb_dim = 32
    time_window_ms = 1000 * 60 * 60 * 1  # 1 hr
    latest_n_emb = 10

    action_type_seq = torch.randint(0, 20, (test_batch_size, full_seq_len))
    item_embedding_seq = torch.rand(test_batch_size, full_seq_len, item_emb_dim)
    action_time_seq = torch.randint(0, 20, (test_batch_size, full_seq_len))
    request_time = torch.randint(500, 1000, (test_batch_size,))
    item_embedding = torch.rand(test_batch_size, item_emb_dim)
    input_features = (
        action_type_seq,
        item_embedding_seq,
        action_time_seq,
        request_time,
        item_embedding,
    )

    print("Initializing TransAct...")
    transact_config = TransActConfig(
        action_vocab=action_vocab,
        seq_len=full_seq_len,
        action_emb_dim=action_emb_dim,
        item_emb_dim=item_emb_dim,
        time_window_ms=time_window_ms,
        latest_n_emb=latest_n_emb,
    )

    transact_module = TransAct(transact_config)

    print("Test forward pass")
    output = transact_module(*input_features)
    print(output)
    print("Test succeeded")


test_run_transact()
