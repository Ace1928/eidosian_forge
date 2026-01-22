import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def optimizer_update_8bit(optimizer_name: str, g: Tensor, p: Tensor, state1: Tensor, state2: Tensor, beta1: float, beta2: float, eps: float, step: int, lr: float, qmap1: Tensor, qmap2: Tensor, max1: Tensor, max2: Tensor, new_max1: Tensor, new_max2: Tensor, weight_decay: float=0.0, gnorm_scale: float=1.0, unorm_vec: Optional[torch.Tensor]=None, max_unorm: float=0.0) -> None:
    """
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer. Choices {adam, momentum}
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Adam state 1.
    state2 : torch.Tensor
        Adam state 2.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    eps : float
        Adam epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    qmap1 : torch.Tensor
        Quantization map for first Adam state.
    qmap2 : torch.Tensor
        Quantization map for second Adam state.
    max1 : torch.Tensor
        Max value for first Adam state update.
    max2 : torch.Tensor
        Max value for second Adam state update.
    new_max1 : torch.Tensor
        Max value for the next Adam update of the first state.
    new_max2 : torch.Tensor
        Max value for the next Adam update of the second state.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    """
    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())
    prev_device = pre_call(g.device)
    is_on_gpu([g, p, state1, state2, unorm_vec, qmap1, qmap2, max1, max2, new_max1, new_max2])
    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][0](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2), get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2), ct.c_float(weight_decay), ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][1](get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2), get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm), ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_int32(step), ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2), get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2), ct.c_float(weight_decay), ct.c_float(gnorm_scale), ct.c_int32(g.numel()))
    else:
        raise ValueError(f'Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}')
    post_call(prev_device)