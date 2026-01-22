from typing import Optional
import torch
import triton
import triton.language as tl
from xformers.triton.k_activations import (
def fused_matmul(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], activation=0, save_act_inputs: bool=False):
    """
    Compute e = activation(x @ weight + bias).
    This wrapper kicks the `kernel_fma` Triton kernel
    """
    if not x.is_contiguous():
        x = x.contiguous()
    x_ = x if x.ndim == 2 else x.flatten(0, -2)
    assert x_.shape[1] == weight.shape[1], f'Incompatible dimensions in between inputs and weight, {x_.shape} - {weight.shape}'
    assert bias is None or bias.is_contiguous()
    assert bias is None or bias.shape[0] == weight.shape[0], 'Incompatible dimensions in between weight and bias'
    assert weight.is_contiguous()
    M, K = x_.shape
    N, K = weight.shape
    outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_inputs = torch.empty_like(outputs) if save_act_inputs else x
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    kernel_fma[grid](outputs, act_inputs, x_, weight, bias if bias is not None else x, M, N, K, outputs.stride(0), x_.stride(0), weight.stride(0), ACTIVATION=activation, BIAS=bias is not None, GROUP_M=8, SAVE_ACT_INPUTS=save_act_inputs, is_fp16=x_.dtype == torch.float16)
    outputs = outputs if x.ndim == 2 else outputs.reshape(*x.shape[:-1], N)
    return (outputs, act_inputs if save_act_inputs else None)