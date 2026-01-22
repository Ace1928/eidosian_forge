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
def test_slice_assignment(self):
    B = self.spcreator((4, 3))
    expected = array([[10, 0, 0], [0, 0, 6], [0, 14, 0], [0, 0, 0]])
    block = [[1, 0], [0, 4]]
    with suppress_warnings() as sup:
        sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
        B[0, 0] = 5
        B[1, 2] = 3
        B[2, 1] = 7
        B[:, :] = B + B
        assert_array_equal(B.toarray(), expected)
        B[:2, :2] = csc_matrix(array(block))
        assert_array_equal(B.toarray()[:2, :2], block)