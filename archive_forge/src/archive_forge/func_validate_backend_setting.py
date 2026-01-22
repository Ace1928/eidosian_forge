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
def validate_backend_setting(name, value, can_use_setenv=True):
    backend_value = getenv(name)
    if value == backend_value or (value == '' and backend_value is None and (platform.system() == 'Windows')):
        return
    if not can_use_setenv:
        raise RuntimeError('Could not set env var {}={} within C Runtime'.format(name, value))
    setenv(name, value)
    validate_backend_setting(name, value, can_use_setenv=False)