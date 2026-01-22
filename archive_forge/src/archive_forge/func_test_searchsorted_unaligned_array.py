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
def test_searchsorted_unaligned_array(self):
    a = np.arange(10)
    aligned = np.empty(a.itemsize * a.size + 1, 'uint8')
    unaligned = aligned[1:].view(a.dtype)
    unaligned[:] = a
    b = unaligned.searchsorted(a, 'left')
    assert_equal(b, a)
    b = unaligned.searchsorted(a, 'right')
    assert_equal(b, a + 1)
    b = a.searchsorted(unaligned, 'left')
    assert_equal(b, a)
    b = a.searchsorted(unaligned, 'right')
    assert_equal(b, a + 1)