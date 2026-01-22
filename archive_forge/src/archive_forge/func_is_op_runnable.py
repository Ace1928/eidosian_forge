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
def is_op_runnable():
    """Returns True for all CPU tests. Returns True for GPU tests that are either of the following.
    1. Built with USE_TVM_OP=0.
    2. Built with USE_TVM_OP=1, but with compute capability >= 53.
    """
    ctx = current_context()
    if ctx.device_type == 'gpu':
        if not _features.is_enabled('TVM_OP'):
            return True
        else:
            try:
                cc = get_cuda_compute_capability(ctx)
            except:
                print('Failed to get CUDA compute capability for context {}. The operators built with USE_TVM_OP=1 will not be run in unit tests.'.format(ctx))
                return False
            print('Cuda arch compute capability: sm_{}'.format(str(cc)))
            return cc >= 53
    return True