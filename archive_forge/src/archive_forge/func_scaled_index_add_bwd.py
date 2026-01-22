from typing import Optional
import torch
import triton
import triton.language as tl
def scaled_index_add_bwd(grad_output: torch.Tensor, grad_source: torch.Tensor, grad_scaling: Optional[torch.Tensor], source: torch.Tensor, scaling: Optional[torch.Tensor], index: torch.Tensor, alpha: float):
    if not (grad_output.is_cuda and grad_source.is_cuda):
        raise ValueError('The grad_output tensor and grad_source tensor must be of type CUDA!')
    if not (grad_output.ndim == 3 and source.ndim == 3):
        raise ValueError(f'The input and source must be three-dimensional (got {grad_output.ndim} and {source.ndim})!')
    if not grad_output.shape[1] == source.shape[1]:
        raise ValueError(f'The number of elements along dimension 1 of the input and source must be the same (got {(grad_output.shape[1],)} and {(source.shape[1],)})!')
    if not grad_output.shape[2] == source.shape[2]:
        raise ValueError(f'The number of elements along dimension 2 of the input and source must be the same (got {(grad_output.shape[2],)} and {(source.shape[2],)})!')
    num_inp_indices, num_rows, num_cols = grad_output.shape
    num_src_indices, num_rows, num_cols = source.shape
    if not num_inp_indices >= num_src_indices:
        raise ValueError(f'The number of elements along dimension 0 of the input must be larger than that of source (got {num_inp_indices} and {num_src_indices})!')
    stride0, stride1, stride2 = (source.stride(0), source.stride(1), source.stride(2))
    if not (grad_output.stride(0) == stride0 and grad_output.stride(1) == stride1 and (grad_output.stride(2) == stride2)):
        raise ValueError(f'The strides of grad_output and source must match (got {grad_output.stride(0)} vs {stride0}, {grad_output.stride(1)} vs {stride1}, {grad_output.stride(2)} vs {stride2})!')
    if not (grad_source.stride(0) == stride0 and grad_source.stride(1) == stride1 and (grad_source.stride(2) == stride2)):
        raise ValueError(f'The strides of grad_source and source must match (got {grad_source.stride(0)} vs {stride0}, {grad_source.stride(1)} vs {stride1}, {grad_source.stride(2)} vs {stride2})!')
    if scaling is not None and grad_scaling is not None:
        HAS_SCALING = True
        if not grad_scaling.is_cuda:
            raise ValueError('The scaling tensor must be of type CUDA!')
        if not (grad_scaling.stride(0) == stride0 and grad_scaling.stride(1) == stride1 and (grad_scaling.stride(2) == stride2)):
            raise ValueError(f'The strides of grad_scaling and source must match (got {grad_scaling.stride(0)} vs {stride0}, {grad_scaling.stride(1)} vs {stride1}, {grad_scaling.stride(2)} vs {stride2})!')
        if not scaling.stride(0) == stride2:
            raise ValueError(f'The stride of scaling must match stride2 of source (got {scaling.stride(0)} vs. {stride2})!')
    else:
        HAS_SCALING = False

    def grid(meta):
        return (triton.cdiv(num_src_indices, meta['BLOCK_SIZE_INDEX']), triton.cdiv(num_rows, meta['BLOCK_SIZE_ROW']), triton.cdiv(num_cols, meta['BLOCK_SIZE_COL']))
    scaled_index_add_bwd_kernel[grid](grad_output, grad_source, grad_scaling, source, scaling, index, alpha, num_inp_indices, num_src_indices, num_rows, num_cols, stride0, stride1, stride2, BLOCK_SIZE_INDEX=1, BLOCK_SIZE_ROW=1, BLOCK_SIZE_COL=512, HAS_SCALING=HAS_SCALING)
    return