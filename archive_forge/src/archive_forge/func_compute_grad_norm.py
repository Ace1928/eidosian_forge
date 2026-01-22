import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def compute_grad_norm(parameters, norm_type=2.0):
    """
    Compute norm over gradients of model parameters.

    :param parameters:
        the model parameters for gradient norm calculation. Iterable of
        Tensors or single Tensor
    :param norm_type:
        type of p-norm to use

    :returns:
        the computed gradient norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None and p.grad is not None]
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)