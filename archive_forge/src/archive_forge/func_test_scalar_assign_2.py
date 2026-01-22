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
def test_scalar_assign_2(self):
    n, m = (5, 10)

    def _test_set(i, j, nitems):
        msg = f'{i!r} ; {j!r} ; {nitems!r}'
        A = self.spcreator((n, m))
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            A[i, j] = 1
        assert_almost_equal(A.sum(), nitems, err_msg=msg)
        assert_almost_equal(A[i, j], 1, err_msg=msg)
    for i, j in [(2, 3), (-1, 8), (-1, -2), (array(-1), -2), (-1, array(-2)), (array(-1), array(-2))]:
        _test_set(i, j, 1)