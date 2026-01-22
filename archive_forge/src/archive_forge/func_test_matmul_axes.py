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
def test_matmul_axes():
    a = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    c = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (1, 2)])
    assert c.shape == (3, 4, 4)
    d = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (0, 1)])
    assert d.shape == (4, 4, 3)
    e = np.swapaxes(d, 0, 2)
    assert_array_equal(e, c)
    f = np.matmul(a, np.arange(3), axes=[(1, 0), 0, 0])
    assert f.shape == (4, 5)