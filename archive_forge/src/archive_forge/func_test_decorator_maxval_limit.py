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
def test_decorator_maxval_limit(self):

    @with_64bit_maxval_limit(maxval_limit=10)
    def check(mat_cls):
        m = mat_cls(np.random.rand(10, 1))
        assert_(self._compare_index_dtype(m, np.int32))
        m = mat_cls(np.random.rand(11, 1))
        assert_(self._compare_index_dtype(m, np.int64))
    for mat_cls in self.MAT_CLASSES:
        check(mat_cls)