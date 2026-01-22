import math
from typing import List, Optional
import torch
from torch import Tensor
from .optimizer import (
def radam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[Tensor], decoupled_weight_decay: bool=False, foreach: Optional[bool]=None, differentiable: bool=False, has_complex: bool=False, *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float):
    """Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """
    if not all((isinstance(t, torch.Tensor) for t in state_steps)):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_radam
    else:
        func = _single_tensor_radam
    func(params, grads, exp_avgs, exp_avg_sqs, state_steps, beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, eps=eps, decoupled_weight_decay=decoupled_weight_decay, differentiable=differentiable, has_complex=has_complex)