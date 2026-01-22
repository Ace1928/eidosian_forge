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
def get_torch_default_device() -> 'torch.device':
    if torch is None:
        raise ValueError('Cannot get default Torch device when Torch is not available.')
    from .backends import get_current_ops
    from .backends.cupy_ops import CupyOps
    from .backends.mps_ops import MPSOps
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        device_id = torch.cuda.current_device()
        return torch.device(f'cuda:{device_id}')
    elif isinstance(ops, MPSOps):
        return torch.device('mps')
    return torch.device('cpu')