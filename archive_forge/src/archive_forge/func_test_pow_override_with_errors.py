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
def test_pow_override_with_errors(self):

    class PowerOnly(np.ndarray):

        def __array_ufunc__(self, ufunc, method, *inputs, **kw):
            if ufunc is not np.power:
                raise NotImplementedError
            return 'POWER!'
    a = np.array(5.0, dtype=np.float64).view(PowerOnly)
    assert_equal(a ** 2.5, 'POWER!')
    with assert_raises(NotImplementedError):
        a ** 0.5
    with assert_raises(NotImplementedError):
        a ** 0
    with assert_raises(NotImplementedError):
        a ** 1
    with assert_raises(NotImplementedError):
        a ** (-1)
    with assert_raises(NotImplementedError):
        a ** 2