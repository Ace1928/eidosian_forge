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
def test_any_where(self):
    a = np.array([[True, False, True], [False, False, False], [True, True, True]])
    wh_full = np.array([[False, True, False], [True, True, True], [False, False, False]])
    wh_middle = np.array([[False], [True], [False]])
    for _ax in [0, None]:
        assert_equal(a.any(axis=_ax, where=wh_middle), np.any(a[wh_middle[:, 0], :], axis=_ax))
        assert_equal(np.any(a, axis=_ax, where=wh_middle), a[wh_middle[:, 0], :].any(axis=_ax))
    assert_equal(a.any(where=wh_full), False)
    assert_equal(np.any(a, where=wh_full), False)
    assert_equal(a.any(where=False), False)
    assert_equal(np.any(a, where=False), False)