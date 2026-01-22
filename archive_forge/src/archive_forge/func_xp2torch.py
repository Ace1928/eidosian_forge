import contextlib
import functools
import inspect
import os
import platform
import random
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
import numpy
from packaging.version import Version
from wasabi import table
from .compat import (
from .compat import mxnet as mx
from .compat import tensorflow as tf
from .compat import torch
from typing import TYPE_CHECKING
from . import types  # noqa: E402
from .types import ArgsKwargs, ArrayXd, FloatsXd, IntsXd, Padded, Ragged  # noqa: E402
def xp2torch(xp_tensor: ArrayXd, requires_grad: bool=False, device: Optional['torch.device']=None) -> 'torch.Tensor':
    """Convert a numpy or cupy tensor to a PyTorch tensor."""
    assert_pytorch_installed()
    if device is None:
        device = get_torch_default_device()
    if hasattr(xp_tensor, 'toDlpack'):
        dlpack_tensor = xp_tensor.toDlpack()
        torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
    elif hasattr(xp_tensor, '__dlpack__'):
        torch_tensor = torch.utils.dlpack.from_dlpack(xp_tensor)
    else:
        torch_tensor = torch.from_numpy(xp_tensor)
    torch_tensor = torch_tensor.to(device)
    if requires_grad:
        torch_tensor.requires_grad_()
    return torch_tensor