import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def with_64bit_maxval_limit(maxval_limit=None, random=False, fixed_dtype=None, downcast_maxval=None, assert_32bit=False):
    """
    Monkeypatch the maxval threshold at which scipy.sparse switches to
    64-bit index arrays, or make it (pseudo-)random.

    """
    if maxval_limit is None:
        maxval_limit = np.int64(10)
    else:
        maxval_limit = np.int64(maxval_limit)
    if assert_32bit:

        def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
            tp = get_index_dtype(arrays, maxval, check_contents)
            assert_equal(np.iinfo(tp).max, np.iinfo(np.int32).max)
            assert_(tp == np.int32 or tp == np.intc)
            return tp
    elif fixed_dtype is not None:

        def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
            return fixed_dtype
    elif random:
        counter = np.random.RandomState(seed=1234)

        def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
            return (np.int32, np.int64)[counter.randint(2)]
    else:

        def new_get_index_dtype(arrays=(), maxval=None, check_contents=False):
            dtype = np.int32
            if maxval is not None:
                if maxval > maxval_limit:
                    dtype = np.int64
            for arr in arrays:
                arr = np.asarray(arr)
                if arr.dtype > np.int32:
                    if check_contents:
                        if arr.size == 0:
                            continue
                        elif np.issubdtype(arr.dtype, np.integer):
                            maxval = arr.max()
                            minval = arr.min()
                            if minval >= -maxval_limit and maxval <= maxval_limit:
                                continue
                    dtype = np.int64
            return dtype
    if downcast_maxval is not None:

        def new_downcast_intp_index(arr):
            if arr.max() > downcast_maxval:
                raise AssertionError('downcast limited')
            return arr.astype(np.intp)

    @decorator
    def deco(func, *a, **kw):
        backup = []
        modules = [scipy.sparse._bsr, scipy.sparse._coo, scipy.sparse._csc, scipy.sparse._csr, scipy.sparse._dia, scipy.sparse._dok, scipy.sparse._lil, scipy.sparse._sputils, scipy.sparse._compressed, scipy.sparse._construct]
        try:
            for mod in modules:
                backup.append((mod, 'get_index_dtype', getattr(mod, 'get_index_dtype', None)))
                setattr(mod, 'get_index_dtype', new_get_index_dtype)
                if downcast_maxval is not None:
                    backup.append((mod, 'downcast_intp_index', getattr(mod, 'downcast_intp_index', None)))
                    setattr(mod, 'downcast_intp_index', new_downcast_intp_index)
            return func(*a, **kw)
        finally:
            for mod, name, oldfunc in backup:
                if oldfunc is not None:
                    setattr(mod, name, oldfunc)
    return deco