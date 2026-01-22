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
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_gh_22683():
    b = 777.68760986
    a = np.array([b] * 10000, dtype=object)
    refc_start = sys.getrefcount(b)
    np.choose(np.zeros(10000, dtype=int), [a], out=a)
    np.choose(np.zeros(10000, dtype=int), [a], out=a)
    refc_end = sys.getrefcount(b)
    assert refc_end - refc_start < 10