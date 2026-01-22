import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
def remove_hook_from_submodules(module: nn.Module):
    """
    Recursively removes all hooks attached on the submodules of a given model.

    Args:
        module (`torch.nn.Module`): The module on which to remove all hooks.
    """
    remove_hook_from_module(module)
    for child in module.children():
        remove_hook_from_submodules(child)