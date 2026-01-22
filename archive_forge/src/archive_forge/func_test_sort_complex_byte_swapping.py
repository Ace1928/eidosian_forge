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
def test_sort_complex_byte_swapping(self):
    for endianness in '<>':
        for dt in np.typecodes['Complex']:
            arr = np.array([1 + 3j, 2 + 2j, 3 + 1j], dtype=endianness + dt)
            c = arr.copy()
            c.sort()
            msg = 'byte-swapped complex sort, dtype={0}'.format(dt)
            assert_equal(c, arr, msg)