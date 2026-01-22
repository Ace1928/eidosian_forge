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
@contextlib.contextmanager
def use_nvtx_range(message: str, id_color: int=-1):
    """Context manager to register the executed code as an NVTX range. The
    ranges can be used as markers in CUDA profiling."""
    if has_cupy:
        cupy.cuda.nvtx.RangePush(message, id_color)
        yield
        cupy.cuda.nvtx.RangePop()
    else:
        yield