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
def test_compatible_cast(self):

    def int_types(byteswap=False):
        int_types = np.typecodes['Integer'] + np.typecodes['UnsignedInteger']
        for int_type in int_types:
            yield np.dtype(int_type)
            if byteswap:
                yield np.dtype(int_type).newbyteorder()
    for int1 in int_types():
        for int2 in int_types(True):
            arr = np.arange(10, dtype=int1)
            for copy in self.true_vals:
                res = np.array(arr, copy=copy, dtype=int2)
                assert res is not arr and res.flags.owndata
                assert_array_equal(res, arr)
            if int1 == int2:
                for copy in self.false_vals:
                    res = np.array(arr, copy=copy, dtype=int2)
                    assert res is arr or res.base is arr
                res = np.array(arr, copy=np._CopyMode.NEVER, dtype=int2)
                assert res is arr or res.base is arr
            else:
                for copy in self.false_vals:
                    res = np.array(arr, copy=copy, dtype=int2)
                    assert res is not arr and res.flags.owndata
                    assert_array_equal(res, arr)
                assert_raises(ValueError, np.array, arr, copy=np._CopyMode.NEVER, dtype=int2)
                assert_raises(ValueError, np.array, arr, copy=None, dtype=int2)