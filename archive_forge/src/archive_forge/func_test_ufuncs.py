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
def test_ufuncs(self):
    X = csc_matrix(np.arange(21).reshape(7, 3) / 21.0)
    for f in ['sin', 'tan', 'arcsin', 'arctan', 'sinh', 'tanh', 'arcsinh', 'arctanh', 'rint', 'sign', 'expm1', 'log1p', 'deg2rad', 'rad2deg', 'floor', 'ceil', 'trunc', 'sqrt']:
        assert_equal(hasattr(csr_matrix, f), True)
        X2 = getattr(X, f)()
        assert_equal(X.shape, X2.shape)
        assert_array_equal(X.indices, X2.indices)
        assert_array_equal(X.indptr, X2.indptr)
        assert_array_equal(X2.toarray(), getattr(np, f)(X.toarray()))