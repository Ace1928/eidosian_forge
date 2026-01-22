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
def test_equal_override():

    class MyAlwaysEqual:

        def __eq__(self, other):
            return 'eq'

        def __ne__(self, other):
            return 'ne'

    class MyAlwaysEqualOld(MyAlwaysEqual):
        __array_priority__ = 10000

    class MyAlwaysEqualNew(MyAlwaysEqual):
        __array_ufunc__ = None
    array = np.array([(0, 1), (2, 3)], dtype='i4,i4')
    for my_always_equal_cls in (MyAlwaysEqualOld, MyAlwaysEqualNew):
        my_always_equal = my_always_equal_cls()
        assert_equal(my_always_equal == array, 'eq')
        assert_equal(array == my_always_equal, 'eq')
        assert_equal(my_always_equal != array, 'ne')
        assert_equal(array != my_always_equal, 'ne')