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
def test_array_scalar_relational_operation(self):
    for dt1 in np.typecodes['AllInteger']:
        assert_(1 > np.array(0, dtype=dt1), 'type %s failed' % (dt1,))
        assert_(not 1 < np.array(0, dtype=dt1), 'type %s failed' % (dt1,))
        for dt2 in np.typecodes['AllInteger']:
            assert_(np.array(1, dtype=dt1) > np.array(0, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))
            assert_(not np.array(1, dtype=dt1) < np.array(0, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))
    for dt1 in 'BHILQP':
        assert_(-1 < np.array(1, dtype=dt1), 'type %s failed' % (dt1,))
        assert_(not -1 > np.array(1, dtype=dt1), 'type %s failed' % (dt1,))
        assert_(-1 != np.array(1, dtype=dt1), 'type %s failed' % (dt1,))
        for dt2 in 'bhilqp':
            assert_(np.array(1, dtype=dt1) > np.array(-1, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))
            assert_(not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))
            assert_(np.array(1, dtype=dt1) != np.array(-1, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))
    for dt1 in 'bhlqp' + np.typecodes['Float']:
        assert_(1 > np.array(-1, dtype=dt1), 'type %s failed' % (dt1,))
        assert_(not 1 < np.array(-1, dtype=dt1), 'type %s failed' % (dt1,))
        assert_(-1 == np.array(-1, dtype=dt1), 'type %s failed' % (dt1,))
        for dt2 in 'bhlqp' + np.typecodes['Float']:
            assert_(np.array(1, dtype=dt1) > np.array(-1, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))
            assert_(not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))
            assert_(np.array(-1, dtype=dt1) == np.array(-1, dtype=dt2), 'type %s and %s failed' % (dt1, dt2))