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
def test_trailing_padding(self):
    align = np.dtype('i').alignment
    size = np.dtype('i').itemsize

    def aligned(n):
        return align * (1 + (n - 1) // align)
    base = dict(formats=['i'], names=['f0'])
    self._check('ix', dict(itemsize=aligned(size + 1), **base))
    self._check('ixx', dict(itemsize=aligned(size + 2), **base))
    self._check('ixxx', dict(itemsize=aligned(size + 3), **base))
    self._check('ixxxx', dict(itemsize=aligned(size + 4), **base))
    self._check('i7x', dict(itemsize=aligned(size + 7), **base))
    self._check('^ix', dict(itemsize=size + 1, **base))
    self._check('^ixx', dict(itemsize=size + 2, **base))
    self._check('^ixxx', dict(itemsize=size + 3, **base))
    self._check('^ixxxx', dict(itemsize=size + 4, **base))
    self._check('^i7x', dict(itemsize=size + 7, **base))