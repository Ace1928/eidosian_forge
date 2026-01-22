import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
@wraps(fn)
def caller(*args, **kwargs):
    ba = sign.bind(*args, **kwargs)
    for name, value in ba.arguments.items():
        if sign.parameters[name].annotation is torch.distributed.ProcessGroup:
            from .._C import box_process_group
            ba.arguments[name] = box_process_group(value)
    return dispatcher_impl(*ba.args, **ba.kwargs)