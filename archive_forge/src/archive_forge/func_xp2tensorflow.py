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
def xp2tensorflow(xp_tensor: ArrayXd, requires_grad: bool=False, as_variable: bool=False) -> 'tf.Tensor':
    """Convert a numpy or cupy tensor to a TensorFlow Tensor or Variable"""
    assert_tensorflow_installed()
    if hasattr(xp_tensor, 'toDlpack'):
        dlpack_tensor = xp_tensor.toDlpack()
        tf_tensor = tf.experimental.dlpack.from_dlpack(dlpack_tensor)
    elif hasattr(xp_tensor, '__dlpack__'):
        dlpack_tensor = xp_tensor.__dlpack__()
        tf_tensor = tf.experimental.dlpack.from_dlpack(dlpack_tensor)
    else:
        tf_tensor = tf.convert_to_tensor(xp_tensor)
    if as_variable:
        with tf.device(tf_tensor.device):
            tf_tensor = tf.Variable(tf_tensor, trainable=requires_grad)
    if requires_grad is False and as_variable is False:
        with tf.device(tf_tensor.device):
            tf_tensor = tf.stop_gradient(tf_tensor)
    return tf_tensor