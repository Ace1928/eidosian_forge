import torch
def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered, compile=False):
    if compile:
        from torch._dynamo.utils import is_compile_supported
        if is_compile_supported(sparse.device.type):
            kernel = torch.compile(_sparse_semi_structured_to_dense_cutlass)
            return kernel(sparse, meta_reordered)
    return _sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered)