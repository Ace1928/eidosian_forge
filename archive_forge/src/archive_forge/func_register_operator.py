import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def register_operator(cls: ClsT) -> ClsT:
    global OPERATORS_REGISTRY, FUNC_TO_XFORMERS_OPERATOR
    OPERATORS_REGISTRY.append(cls)
    FUNC_TO_XFORMERS_OPERATOR[cls.OPERATOR] = cls
    return cls