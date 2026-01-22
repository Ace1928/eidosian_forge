import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def simple_forward(sym, ctx=None, is_train=False, **inputs):
    """A simple forward function for a symbol.

    Primarily used in doctest to test the functionality of a symbol.
    Takes NumPy arrays as inputs and outputs are also converted to NumPy arrays.

    Parameters
    ----------
    ctx : Context
        If ``None``, will take the default context.
    inputs : keyword arguments
        Mapping each input name to a NumPy array.

    Returns
    -------
    The result as a numpy array. Multiple results will
    be returned as a list of NumPy arrays.
    """
    ctx = ctx or default_context()
    inputs = {k: array(v) for k, v in inputs.items()}
    exe = sym.bind(ctx, args=inputs)
    exe.forward(is_train=is_train)
    outputs = [x.asnumpy() for x in exe.outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs