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
def test_mapiter(self):
    a = np.arange(12).reshape((3, 4)).astype(float)
    index = ([1, 1, 2, 0], [0, 0, 2, 3])
    vals = [50, 50, 30, 16]
    _multiarray_tests.test_inplace_increment(a, index, vals)
    assert_equal(a, [[0.0, 1.0, 2.0, 19.0], [104.0, 5.0, 6.0, 7.0], [8.0, 9.0, 40.0, 11.0]])
    b = np.arange(6).astype(float)
    index = (np.array([1, 2, 0]),)
    vals = [50, 4, 100.1]
    _multiarray_tests.test_inplace_increment(b, index, vals)
    assert_equal(b, [100.1, 51.0, 6.0, 3.0, 4.0, 5.0])