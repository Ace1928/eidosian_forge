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
@pytest.mark.parametrize('op', [operator.eq, operator.ne])
@pytest.mark.parametrize(['dt1', 'dt2'], [([('f', 'i')], [('f', 'i')]), ('M8', 'd'), ('d', 'd')])
def test_equal_subclass_no_override(op, dt1, dt2):

    class MyArr(np.ndarray):
        called_wrap = 0

        def __array_wrap__(self, new):
            type(self).called_wrap += 1
            return super().__array_wrap__(new)
    numpy_arr = np.zeros(5, dtype=dt1)
    my_arr = np.zeros(5, dtype=dt2).view(MyArr)
    assert type(op(numpy_arr, my_arr)) is MyArr
    assert type(op(my_arr, numpy_arr)) is MyArr
    assert MyArr.called_wrap == 2