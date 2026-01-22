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
def test_pow_array_object_dtype(self):

    class SomeClass:

        def __init__(self, num=None):
            self.num = num

        def __mul__(self, other):
            raise AssertionError('__mul__ should not be called')

        def __div__(self, other):
            raise AssertionError('__div__ should not be called')

        def __pow__(self, exp):
            return SomeClass(num=self.num ** exp)

        def __eq__(self, other):
            if isinstance(other, SomeClass):
                return self.num == other.num
        __rpow__ = __pow__

    def pow_for(exp, arr):
        return np.array([x ** exp for x in arr])
    obj_arr = np.array([SomeClass(1), SomeClass(2), SomeClass(3)])
    assert_equal(obj_arr ** 0.5, pow_for(0.5, obj_arr))
    assert_equal(obj_arr ** 0, pow_for(0, obj_arr))
    assert_equal(obj_arr ** 1, pow_for(1, obj_arr))
    assert_equal(obj_arr ** (-1), pow_for(-1, obj_arr))
    assert_equal(obj_arr ** 2, pow_for(2, obj_arr))