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
def test_structured_comparisons_with_promotion(self):
    a = np.array([(5, 42), (10, 1)], dtype=[('a', '>i8'), ('b', '<f8')])
    b = np.array([(5, 43), (10, 1)], dtype=[('a', '<i8'), ('b', '>f8')])
    assert_equal(a == b, [False, True])
    assert_equal(a != b, [True, False])
    a = np.array([(5, 42), (10, 1)], dtype=[('a', '>f8'), ('b', '<f8')])
    b = np.array([(5, 43), (10, 1)], dtype=[('a', '<i8'), ('b', '>i8')])
    assert_equal(a == b, [False, True])
    assert_equal(a != b, [True, False])
    a = np.array([(5, 42), (10, 1)], dtype=[('a', '10>f8'), ('b', '5<f8')])
    b = np.array([(5, 43), (10, 1)], dtype=[('a', '10<i8'), ('b', '5>i8')])
    assert_equal(a == b, [False, True])
    assert_equal(a != b, [True, False])