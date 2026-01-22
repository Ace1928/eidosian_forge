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
def rand_ndarray(shape, stype='default', density=None, dtype=None, modifier_func=None, shuffle_csr_indices=False, distribution=None, ctx=None):
    """Generate a random sparse ndarray. Returns the generated ndarray."""
    ctx = ctx if ctx else default_context()
    if stype == 'default':
        arr = mx.nd.array(random_arrays(shape), dtype=dtype, ctx=ctx)
    else:
        arr, _ = rand_sparse_ndarray(shape, stype, density=density, modifier_func=modifier_func, dtype=dtype, shuffle_csr_indices=shuffle_csr_indices, distribution=distribution, ctx=ctx)
    return arr