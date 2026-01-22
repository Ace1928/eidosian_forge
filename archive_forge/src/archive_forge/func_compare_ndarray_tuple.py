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
def compare_ndarray_tuple(t1, t2, rtol=None, atol=None):
    """Compare ndarray tuple."""
    if t1 is None or t2 is None:
        return
    if isinstance(t1, tuple):
        for s1, s2 in zip(t1, t2):
            compare_ndarray_tuple(s1, s2, rtol, atol)
    else:
        assert_almost_equal(t1, t2, rtol=rtol, atol=atol)