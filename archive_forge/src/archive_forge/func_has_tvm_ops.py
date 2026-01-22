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
def has_tvm_ops():
    """Returns True if MXNet is compiled with TVM generated operators. If current ctx
    is GPU, it only returns True for CUDA compute capability > 52 where FP16 is supported.
    """
    built_with_tvm_op = _features.is_enabled('TVM_OP')
    ctx = current_context()
    if ctx.device_type == 'gpu':
        try:
            cc = get_cuda_compute_capability(ctx)
        except:
            print('Failed to get CUDA compute capability for context {}. The operators built with USE_TVM_OP=1 will not be run in unit tests.'.format(ctx))
            return False
        print('Cuda arch compute capability: sm_{}'.format(str(cc)))
        return built_with_tvm_op and cc >= 53
    return built_with_tvm_op