from __future__ import annotations
import copy
from typing import Optional, Tuple, TypeVar
import torch
def fuse_linear_bn_eval(linear: LinearT, bn: torch.nn.modules.batchnorm._BatchNorm) -> LinearT:
    """Fuse a linear module and a BatchNorm module into a single, new linear module.

    Args:
        linear (torch.nn.Linear): A Linear module.
        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.

    Returns:
        torch.nn.Linear: The fused linear module.

    .. note::
        Both ``linear`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.
    """
    assert not (linear.training or bn.training), 'Fusion only for eval!'
    fused_linear = copy.deepcopy(linear)
    assert bn.running_mean is not None and bn.running_var is not None
    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(fused_linear.weight, fused_linear.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_linear