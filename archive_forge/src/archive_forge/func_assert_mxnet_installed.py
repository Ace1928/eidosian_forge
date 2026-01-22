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
def assert_mxnet_installed() -> None:
    """Raise an ImportError if MXNet is not installed."""
    if not has_mxnet:
        raise ImportError('MXNet support requires mxnet: pip install thinc[mxnet]\n\nEnable MXNet support with thinc.api.enable_mxnet()')