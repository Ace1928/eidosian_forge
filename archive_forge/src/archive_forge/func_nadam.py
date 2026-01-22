import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt, _stack_if_compiling,
from typing import List, Optional
def nadam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], mu_products: List[Tensor], state_steps: List[Tensor], decoupled_weight_decay: bool=False, foreach: Optional[bool]=None, capturable: bool=False, differentiable: bool=False, has_complex: bool=False, *, beta1: float, beta2: float, lr: float, weight_decay: float, momentum_decay: float, eps: float):
    """Functional API that performs NAdam algorithm computation.

    See :class:`~torch.optim.NAdam` for details.
    """
    if not all((isinstance(t, torch.Tensor) for t in state_steps)):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if not all((isinstance(t, torch.Tensor) for t in mu_products)):
        raise RuntimeError('API has changed, `mu_products` argument must contain a list of singleton tensors')
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_nadam
    else:
        func = _single_tensor_nadam
    func(params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps, beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, momentum_decay=momentum_decay, decoupled_weight_decay=decoupled_weight_decay, eps=eps, capturable=capturable, differentiable=differentiable, has_complex=has_complex)