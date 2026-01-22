import inspect
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
@register_decomposition_for_jvp(aten.native_layer_norm_backward)
def native_layer_norm_backward(grad_out: Tensor, input: Tensor, normalized_shape: List[int], mean: Tensor, rstd: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], output_mask: List[bool]) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_ndim = input.dim()
    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices = list(range(axis, input_ndim))
    outer_dim_indices = list(range(0, axis))
    N = 1
    for i in inner_dims:
        N *= i
    M = 1
    for i in outer_dims:
        M *= i
    if M <= 0 or N <= 0:
        return (input.new_zeros(input_shape), input.new_zeros(input_shape[axis:]), input.new_zeros(input_shape[axis:]))
    mean_, rstd_ = recompute_mean_var(input, rstd, inner_dim_indices, keepdim=True)
    x_hat = (input - mean_) * rstd_
    if weight is not None:
        grad_x_hat = grad_out * weight
    else:
        grad_x_hat = grad_out
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)
    inner = a - b - c3
    if output_mask[0]:
        d_input: Optional[Tensor] = rstd_ / N * inner
    else:
        d_input = torch.zeros_like(input)
    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            d_weight: Optional[Tensor] = torch.sum(grad_out * x_hat, outer_dim_indices, False)
        else:
            d_weight = grad_out * x_hat
    elif weight is not None:
        d_weight = torch.zeros_like(weight)
    else:
        d_weight = torch.zeros(())
    if output_mask[2] and bias is not None:
        if len(outer_dim_indices) > 0:
            d_bias: Optional[Tensor] = torch.sum(grad_out, outer_dim_indices, False)
        else:
            d_bias = grad_out.clone()
    elif bias is not None:
        d_bias = torch.zeros_like(bias)
    else:
        d_bias = torch.zeros(())
    return (d_input, d_weight, d_bias)