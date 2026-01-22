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
def test_multifield_indexing_view(self):
    a = np.ones(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u4')])
    v = a[['a', 'c']]
    assert_(v.base is a)
    assert_(v.dtype == np.dtype({'names': ['a', 'c'], 'formats': ['i4', 'u4'], 'offsets': [0, 8]}))
    v[:] = (4, 5)
    assert_equal(a[0].item(), (4, 1, 5))