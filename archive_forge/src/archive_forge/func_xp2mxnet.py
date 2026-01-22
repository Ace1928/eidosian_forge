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
def xp2mxnet(xp_tensor: ArrayXd, requires_grad: bool=False) -> 'mx.nd.NDArray':
    """Convert a numpy or cupy tensor to a MXNet tensor."""
    assert_mxnet_installed()
    if hasattr(xp_tensor, 'toDlpack'):
        dlpack_tensor = xp_tensor.toDlpack()
        mx_tensor = mx.nd.from_dlpack(dlpack_tensor)
    else:
        mx_tensor = mx.nd.from_numpy(xp_tensor)
    if requires_grad:
        mx_tensor.attach_grad()
    return mx_tensor