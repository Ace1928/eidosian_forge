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
def test_fancy_indexing_broadcast(self):
    I = np.array([[1], [2], [3]])
    J = np.array([3, 4, 2])
    np.random.seed(1234)
    D = asmatrix(np.random.rand(5, 7))
    S = self.spcreator(D)
    SIJ = S[I, J]
    if issparse(SIJ):
        SIJ = SIJ.toarray()
    assert_equal(SIJ, D[I, J])