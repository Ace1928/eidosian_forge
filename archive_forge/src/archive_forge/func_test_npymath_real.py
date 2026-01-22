from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
def test_npymath_real():
    from numpy.core._multiarray_tests import npy_log10, npy_cosh, npy_sinh, npy_tan, npy_tanh
    funcs = {npy_log10: np.log10, npy_cosh: np.cosh, npy_sinh: np.sinh, npy_tan: np.tan, npy_tanh: np.tanh}
    vals = (1, np.inf, -np.inf, np.nan)
    types = (np.float32, np.float64, np.longdouble)
    with np.errstate(all='ignore'):
        for fun, npfun in funcs.items():
            for x, t in itertools.product(vals, types):
                z = t(x)
                got = fun(z)
                expected = npfun(z)
                assert_allclose(got, expected)