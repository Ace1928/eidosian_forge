import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def make_pytorch_cuda_operator(fn: ClsT) -> ClsT:
    return turn_into_pytorch_op(fn, 'CUDA')