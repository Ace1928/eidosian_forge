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
@pytest.mark.parametrize('arr', [np.ones(()), np.arange(81).reshape((9, 9))])
@pytest.mark.parametrize('order1', ['C', 'F', None])
@pytest.mark.parametrize('order2', ['C', 'F', 'A', 'K'])
def test_order_mismatch(self, arr, order1, order2):
    arr = arr.copy(order1)
    if order1 == 'C':
        assert arr.flags.c_contiguous
    elif order1 == 'F':
        assert arr.flags.f_contiguous
    elif arr.ndim != 0:
        arr = arr[::2, ::2]
        assert not arr.flags.forc
    if order2 == 'C':
        no_copy_necessary = arr.flags.c_contiguous
    elif order2 == 'F':
        no_copy_necessary = arr.flags.f_contiguous
    else:
        no_copy_necessary = True
    for view in [arr, memoryview(arr)]:
        for copy in self.true_vals:
            res = np.array(view, copy=copy, order=order2)
            assert res is not arr and res.flags.owndata
            assert_array_equal(arr, res)
        if no_copy_necessary:
            for copy in self.false_vals:
                res = np.array(view, copy=copy, order=order2)
                if not IS_PYPY:
                    assert res is arr or res.base.obj is arr
            res = np.array(view, copy=np._CopyMode.NEVER, order=order2)
            if not IS_PYPY:
                assert res is arr or res.base.obj is arr
        else:
            for copy in self.false_vals:
                res = np.array(arr, copy=copy, order=order2)
                assert_array_equal(arr, res)
            assert_raises(ValueError, np.array, view, copy=np._CopyMode.NEVER, order=order2)
            assert_raises(ValueError, np.array, view, copy=None, order=order2)