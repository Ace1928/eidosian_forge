import torch
import triton
import triton.language as tl
def index_select_cat_bwd(grad_source: torch.Tensor, index: torch.Tensor, grad_output: torch.Tensor):
    if not (grad_source.is_cuda and grad_output.is_cuda):
        raise ValueError('The grad_source and grad_output tensor must be of type CUDA!')
    if not (grad_source.ndim == 2 and grad_output.ndim == 2):
        raise ValueError(f'The grad_source and grad_output must be three-dimensional (got {grad_source.ndim} and {grad_output.ndim})!')
    if not grad_source.shape[1] == grad_output.shape[1]:
        raise ValueError(f'The number of elements along dimension 1 of grad_source and grad_output must be the same (got {grad_source.shape[1]} and {grad_output.shape[1]})')
    num_rows, num_cols = grad_source.shape
    num_indices, num_cols = grad_output.shape
    if not num_rows >= num_indices:
        raise ValueError(f'The number of elements along dimension 0 of grad_source must be larger than that of grad_output (got {num_rows} and {num_indices})!')
    if not index.shape[0] == num_indices:
        raise ValueError(f'The number of indices and the number of elements along dimension 0 of grad_output must match (got {index.shape[0]} and {num_indices})!')
    stride0, stride1 = (grad_source.stride(0), grad_source.stride(1))
    if not (grad_output.stride(0) == stride0 and grad_output.stride(1) == stride1):
        raise ValueError(f'The strides of the grad_source and grad_output tensors must match (got {stride0} vs. {grad_output.stride(0)}, {stride1} vs. {grad_output.stride(1)})!')

    def grid(meta):
        return (triton.cdiv(num_indices, meta['BLOCK_SIZE_INDEX']), triton.cdiv(num_cols, meta['BLOCK_SIZE_COL']))
    index_select_cat_bwd_kernel[grid](grad_source, index, grad_output, num_rows, num_indices, num_cols, grad_source.stride(0), grad_source.stride(1), BLOCK_SIZE_INDEX=1, BLOCK_SIZE_COL=512)
    return