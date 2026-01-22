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
def create_sparse_array(shape, stype, data_init=None, rsp_indices=None, dtype=None, modifier_func=None, density=0.5, shuffle_csr_indices=False):
    """Create a sparse array, For Rsp, assure indices are in a canonical format"""
    if stype == 'row_sparse':
        if rsp_indices is not None:
            arr_indices = np.asarray(rsp_indices)
            arr_indices.sort()
        else:
            arr_indices = None
        arr_data, (_, _) = rand_sparse_ndarray(shape, stype, density=density, data_init=data_init, rsp_indices=arr_indices, dtype=dtype, modifier_func=modifier_func)
    elif stype == 'csr':
        arr_data, (_, _, _) = rand_sparse_ndarray(shape, stype, density=density, data_init=data_init, dtype=dtype, modifier_func=modifier_func, shuffle_csr_indices=shuffle_csr_indices)
    else:
        msg = 'Unknown storage type: ' + stype
        raise AssertionError(msg)
    return arr_data