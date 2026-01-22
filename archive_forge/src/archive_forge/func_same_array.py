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
def same_array(array1, array2):
    """Check whether two NDArrays sharing the same memory block

    Parameters
    ----------

    array1 : NDArray
        First NDArray to be checked
    array2 : NDArray
        Second NDArray to be checked

    Returns
    -------
    bool
        Whether two NDArrays share the same memory
    """
    array1[:] += 1
    if not same(array1.asnumpy(), array2.asnumpy()):
        array1[:] -= 1
        return False
    array1[:] -= 1
    return same(array1.asnumpy(), array2.asnumpy())