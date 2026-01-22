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
def test_fromfile_offset(self, x, tmp_filename):
    with open(tmp_filename, 'wb') as f:
        x.tofile(f)
    with open(tmp_filename, 'rb') as f:
        y = np.fromfile(f, dtype=x.dtype, offset=0)
        assert_array_equal(y, x.flat)
    with open(tmp_filename, 'rb') as f:
        count_items = len(x.flat) // 8
        offset_items = len(x.flat) // 4
        offset_bytes = x.dtype.itemsize * offset_items
        y = np.fromfile(f, dtype=x.dtype, count=count_items, offset=offset_bytes)
        assert_array_equal(y, x.flat[offset_items:offset_items + count_items])
        offset_bytes = x.dtype.itemsize
        z = np.fromfile(f, dtype=x.dtype, offset=offset_bytes)
        assert_array_equal(z, x.flat[offset_items + count_items + 1:])
    with open(tmp_filename, 'wb') as f:
        x.tofile(f, sep=',')
    with open(tmp_filename, 'rb') as f:
        assert_raises_regex(TypeError, "'offset' argument only permitted for binary files", np.fromfile, tmp_filename, dtype=x.dtype, sep=',', offset=1)