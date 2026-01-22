import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def callee(*args, **kwargs):
    ba = sign.bind(*args, **kwargs)
    for name, value in ba.arguments.items():
        if sign.parameters[name].annotation is torch.distributed.ProcessGroup:
            from .._C import unbox_process_group
            ba.arguments[name] = unbox_process_group(value)
    return fn(*ba.args, **ba.kwargs)