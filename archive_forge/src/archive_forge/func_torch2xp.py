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
def torch2xp(torch_tensor: 'torch.Tensor', *, ops: Optional['Ops']=None) -> ArrayXd:
    """Convert a torch tensor to a numpy or cupy tensor depending on the `ops` parameter.
    If `ops` is `None`, the type of the resultant tensor will be determined by the source tensor's device.
    """
    from .api import NumpyOps
    assert_pytorch_installed()
    if is_torch_cuda_array(torch_tensor):
        if isinstance(ops, NumpyOps):
            return torch_tensor.detach().cpu().numpy()
        else:
            return cupy_from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    elif isinstance(ops, NumpyOps) or ops is None:
        return torch_tensor.detach().cpu().numpy()
    else:
        return cupy.asarray(torch_tensor)