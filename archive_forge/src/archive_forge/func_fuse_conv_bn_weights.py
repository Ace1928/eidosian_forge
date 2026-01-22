from __future__ import annotations
import copy
from typing import Optional, Tuple, TypeVar
import torch
def fuse_conv_bn_weights(conv_w: torch.Tensor, conv_b: Optional[torch.Tensor], bn_rm: torch.Tensor, bn_rv: torch.Tensor, bn_eps: float, bn_w: Optional[torch.Tensor], bn_b: Optional[torch.Tensor], transpose: bool=False) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

    Args:
        conv_w (torch.Tensor): Convolutional weight.
        conv_b (Optional[torch.Tensor]): Convolutional bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (Optional[torch.Tensor]): BatchNorm weight.
        bn_b (Optional[torch.Tensor]): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused convolutional weight and bias.
    """
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)
    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(dtype=conv_weight_dtype)
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(dtype=conv_bias_dtype)
    return (torch.nn.Parameter(fused_conv_w, conv_w.requires_grad), torch.nn.Parameter(fused_conv_b, conv_b.requires_grad))