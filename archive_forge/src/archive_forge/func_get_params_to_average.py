import itertools
from typing import Union, Iterable, Dict, Iterator
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, group
def get_params_to_average(params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]):
    """
    Returns a list of parameters that need to average, which filters out the parameters that do not contain any gradients.
    Args:
        params: The parameters of a model or parameter groups of an optimizer.
    """
    filtered_params = []
    for param in params:
        if isinstance(param, torch.nn.Parameter):
            param_data = param
            if param_data.grad is not None:
                filtered_params.append(param_data)
        elif isinstance(param, dict):
            for param_data in param['params']:
                if param_data.grad is not None:
                    filtered_params.append(param_data)
        else:
            raise NotImplementedError(f'Parameter input of type {type(param)} is not supported')
    return filtered_params