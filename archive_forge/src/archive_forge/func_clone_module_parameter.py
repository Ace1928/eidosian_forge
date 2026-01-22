import builtins
import torch
from torch.distributed._shard.sharding_spec import (
from torch.distributed._shard.sharding_spec._internals import (
def clone_module_parameter(module, param_name):
    """
    Clone a parameter from a given existing module.

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter needs to be cloned.
        param_name (str): Name of the parameter of ``module`` that needs to be cloned.

    Returns: cloned tensor as :class:`torch.nn.Parameter`.
    """
    tensor = getattr(module, param_name)
    return torch.nn.Parameter(tensor.detach().clone())