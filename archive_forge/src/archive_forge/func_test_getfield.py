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
def test_getfield():
    a = np.arange(32, dtype='uint16')
    if sys.byteorder == 'little':
        i = 0
        j = 1
    else:
        i = 1
        j = 0
    b = a.getfield('int8', i)
    assert_equal(b, a)
    b = a.getfield('int8', j)
    assert_equal(b, 0)
    pytest.raises(ValueError, a.getfield, 'uint8', -1)
    pytest.raises(ValueError, a.getfield, 'uint8', 16)
    pytest.raises(ValueError, a.getfield, 'uint64', 0)