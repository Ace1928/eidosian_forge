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
def test_export_record(self):
    dt = [('a', 'b'), ('b', 'h'), ('c', 'i'), ('d', 'l'), ('dx', 'q'), ('e', 'B'), ('f', 'H'), ('g', 'I'), ('h', 'L'), ('hx', 'Q'), ('i', np.single), ('j', np.double), ('k', np.longdouble), ('ix', np.csingle), ('jx', np.cdouble), ('kx', np.clongdouble), ('l', 'S4'), ('m', 'U4'), ('n', 'V3'), ('o', '?'), ('p', np.half)]
    x = np.array([(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, b'aaaa', 'bbbb', b'   ', True, 1.0)], dtype=dt)
    y = memoryview(x)
    assert_equal(y.shape, (1,))
    assert_equal(y.ndim, 1)
    assert_equal(y.suboffsets, ())
    sz = sum([np.dtype(b).itemsize for a, b in dt])
    if np.dtype('l').itemsize == 4:
        assert_equal(y.format, 'T{b:a:=h:b:i:c:l:d:q:dx:B:e:@H:f:=I:g:L:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
    else:
        assert_equal(y.format, 'T{b:a:=h:b:i:c:q:d:q:dx:B:e:@H:f:=I:g:Q:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
    if not np.ones(1).strides[0] == np.iinfo(np.intp).max:
        assert_equal(y.strides, (sz,))
    assert_equal(y.itemsize, sz)