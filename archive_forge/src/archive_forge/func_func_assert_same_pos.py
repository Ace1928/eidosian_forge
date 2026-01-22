import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def func_assert_same_pos(x, y, func=isnan, hasval='nan'):
    """Handling nan/inf.

        Combine results of running func on x and y, checking that they are True
        at the same locations.

        """
    __tracebackhide__ = True
    x_id = func(x)
    y_id = func(y)
    if (x_id == y_id).all().item() is not True:
        msg = build_err_msg([x, y], err_msg + '\nx and y %s location mismatch:' % hasval, verbose=verbose, header=header, names=('x', 'y'), precision=precision)
        raise AssertionError(msg)
    if isinstance(x_id, bool) or x_id.ndim == 0:
        return bool_(x_id)
    elif isinstance(y_id, bool) or y_id.ndim == 0:
        return bool_(y_id)
    else:
        return y_id