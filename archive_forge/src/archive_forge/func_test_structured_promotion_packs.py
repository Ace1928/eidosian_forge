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
@pytest.mark.parametrize('dtype_dict', [dict(names=['a', 'b'], formats=['i4', 'f'], itemsize=100), dict(names=['a', 'b'], formats=['i4', 'f'], offsets=[0, 12])])
@pytest.mark.parametrize('align', [True, False])
def test_structured_promotion_packs(self, dtype_dict, align):
    dtype = np.dtype(dtype_dict, align=align)
    dtype_dict.pop('itemsize', None)
    dtype_dict.pop('offsets', None)
    expected = np.dtype(dtype_dict, align=align)
    res = np.promote_types(dtype, dtype)
    assert res.itemsize == expected.itemsize
    assert res.fields == expected.fields
    res = np.promote_types(expected, expected)
    assert res is expected