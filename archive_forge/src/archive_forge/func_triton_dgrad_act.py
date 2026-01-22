from typing import Optional
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
from flash_attn.ops.triton.k_activations import (
def triton_dgrad_act(grad_output: torch.Tensor, weight: torch.Tensor, activation: str='id', act_input: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Compute e = activation(grad_output @ weight + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param grad_output: input tensor
    :param weight: weight matrix
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    assert activation in ['id', 'gelu', 'gelu_approx', 'squared_relu']
    batch_shape, n = (grad_output.shape[:-1], grad_output.shape[-1])
    batch_dim = batch_shape.numel()
    grad_output_reshaped = grad_output.reshape(batch_dim, n)
    if grad_output_reshaped.stride(0) > 1 and grad_output_reshaped.stride(1) > 1:
        grad_output_reshaped = grad_output_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    assert grad_output.dtype == weight.dtype, f'grad_output and weight must have the same dtype, got {grad_output.dtype} and {weight.dtype}'
    assert grad_output_reshaped.shape[1] == weight.shape[0], f'Incompatible dimensions: {grad_output_reshaped.shape} - {weight.shape}'
    if activation != 'id':
        assert act_input is not None, f'act_input is required for activation {activation}'
    M, K = grad_output_reshaped.shape
    K, N = weight.shape
    grad_input = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    kernel_bwd[grid](grad_input, act_input, grad_output_reshaped, weight, M, N, K, M // 32, N // 32, K // 32, stride_cm=grad_input.stride(0), stride_am=grad_output_reshaped.stride(0), stride_ak=grad_output_reshaped.stride(1), stride_bk=weight.stride(0), stride_bn=weight.stride(1), ACTIVATION=activation, GROUP_M=8)
    return grad_input.reshape(*batch_shape, grad_input.shape[-1])