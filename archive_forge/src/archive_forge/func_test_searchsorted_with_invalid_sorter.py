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
def test_searchsorted_with_invalid_sorter(self):
    a = np.array([5, 2, 1, 3, 4])
    s = np.argsort(a)
    assert_raises(TypeError, np.searchsorted, a, 0, sorter=np.array((1, (2, 3)), dtype=object))
    assert_raises(TypeError, np.searchsorted, a, 0, sorter=[1.1])
    assert_raises(ValueError, np.searchsorted, a, 0, sorter=[1, 2, 3, 4])
    assert_raises(ValueError, np.searchsorted, a, 0, sorter=[1, 2, 3, 4, 5, 6])
    assert_raises(ValueError, np.searchsorted, a, 4, sorter=[0, 1, 2, 3, 5])
    assert_raises(ValueError, np.searchsorted, a, 0, sorter=[-1, 0, 1, 2, 3])
    assert_raises(ValueError, np.searchsorted, a, 0, sorter=[4, 0, -1, 2, 3])