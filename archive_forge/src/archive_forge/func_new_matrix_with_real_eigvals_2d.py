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
def new_matrix_with_real_eigvals_2d(n):
    """Generate a well-conditioned matrix with small real eigenvalues."""
    shape = (n, n)
    q = np.ones(shape)
    while 1:
        D = np.diag(np.random.uniform(-1.0, 1.0, shape[-1]))
        I = np.eye(shape[-1]).reshape(shape)
        v = np.random.uniform(-1.0, 1.0, shape[-1]).reshape(shape[:-1] + (1,))
        v = v / np.linalg.norm(v, axis=-2, keepdims=True)
        v_T = np.swapaxes(v, -1, -2)
        U = I - 2 * np.matmul(v, v_T)
        q = np.matmul(U, D)
        if np.linalg.cond(q, 2) < 3:
            break
    D = np.diag(np.random.uniform(-10.0, 10.0, n))
    q_inv = np.linalg.inv(q)
    return np.matmul(np.matmul(q_inv, D), q)