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
def test_datetime64_byteorder(self):
    original = np.array([['2015-02-24T00:00:00.000000000']], dtype='datetime64[ns]')
    original_byte_reversed = original.copy(order='K')
    original_byte_reversed.dtype = original_byte_reversed.dtype.newbyteorder('S')
    original_byte_reversed.byteswap(inplace=True)
    new = pickle.loads(pickle.dumps(original_byte_reversed))
    assert_equal(original.dtype, new.dtype)