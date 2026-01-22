import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
from typing import List, Optional
def rprop(params: List[Tensor], grads: List[Tensor], prevs: List[Tensor], step_sizes: List[Tensor], foreach: Optional[bool]=None, maximize: bool=False, differentiable: bool=False, has_complex: bool=False, *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float):
    """Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_rprop
    else:
        func = _single_tensor_rprop
    func(params, grads, prevs, step_sizes, step_size_min=step_size_min, step_size_max=step_size_max, etaminus=etaminus, etaplus=etaplus, maximize=maximize, differentiable=differentiable, has_complex=has_complex)