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
def test_setfield(self):
    struct_dt = np.dtype([('elem', 'i4', 5)])
    dt = np.dtype([('field', 'i4', 10), ('struct', struct_dt)])
    x = np.zeros(1, dt)
    x[0]['field'] = np.ones(10, dtype='i4')
    x[0]['struct'] = np.ones(1, dtype=struct_dt)
    assert_equal(x[0]['field'], np.ones(10, dtype='i4'))