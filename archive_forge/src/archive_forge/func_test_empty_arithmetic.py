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
def test_empty_arithmetic(self):
    shape = (5, 5)
    for mytype in [np.dtype('int32'), np.dtype('float32'), np.dtype('float64'), np.dtype('complex64'), np.dtype('complex128')]:
        a = self.spcreator(shape, dtype=mytype)
        b = a + a
        c = 2 * a
        d = a @ a.tocsc()
        e = a @ a.tocsr()
        f = a @ a.tocoo()
        for m in [a, b, c, d, e, f]:
            assert_equal(m.toarray(), a.toarray() @ a.toarray())
            assert_equal(m.dtype, mytype)
            assert_equal(m.toarray().dtype, mytype)