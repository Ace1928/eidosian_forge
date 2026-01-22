import torch
from typing import Iterable, Optional
def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    """Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        num_param = param.numel()
        param.data = vec[pointer:pointer + num_param].view_as(param).data
        pointer += num_param