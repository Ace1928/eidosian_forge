def get_dynamic_sparse_quantized_mapping():
    import torch.ao.nn.sparse
    _dynamic_sparse_quantized_mapping = {torch.nn.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear}
    return _dynamic_sparse_quantized_mapping