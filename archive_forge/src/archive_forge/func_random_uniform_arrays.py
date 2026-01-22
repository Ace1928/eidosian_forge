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
def random_uniform_arrays(*shapes, **kwargs):
    """Generate some random numpy arrays."""
    low = kwargs.pop('low', 0.0)
    high = kwargs.pop('high', 1.0)
    dtype = kwargs.pop('dtype', default_dtype())
    if len(kwargs) > 0:
        raise TypeError('Got unexpected argument/s : ' + str(kwargs.keys()))
    arrays = [np.random.uniform(low, high, size=s).astype(dtype) for s in shapes]
    return arrays