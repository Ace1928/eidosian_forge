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
@pytest.mark.parametrize(['dt1', 'dt2'], [('M8[ns]', 'd'), ('M8[s]', 'l'), ('m8[ns]', 'd'), ('M8[s]', 'm8[s]'), ('S5', 'U5')])
def test_no_loop_gives_all_true_or_false(dt1, dt2):
    arr1 = np.random.randint(5, size=100).astype(dt1)
    arr2 = np.random.randint(5, size=99)[:, np.newaxis].astype(dt2)
    res = arr1 == arr2
    assert res.shape == (99, 100)
    assert res.dtype == bool
    assert not res.any()
    res = arr1 != arr2
    assert res.shape == (99, 100)
    assert res.dtype == bool
    assert res.all()
    arr2 = np.random.randint(5, size=99).astype(dt2)
    with pytest.raises(ValueError):
        arr1 == arr2
    with pytest.raises(ValueError):
        arr1 != arr2
    with pytest.raises(np.core._exceptions._UFuncNoLoopError):
        arr1 > arr2