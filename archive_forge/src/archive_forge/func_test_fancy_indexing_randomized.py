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
def test_fancy_indexing_randomized(self):
    np.random.seed(1234)
    NUM_SAMPLES = 50
    M = 6
    N = 4
    D = asmatrix(np.random.rand(M, N))
    D = np.multiply(D, D > 0.5)
    I = np.random.randint(-M + 1, M, size=NUM_SAMPLES)
    J = np.random.randint(-N + 1, N, size=NUM_SAMPLES)
    S = self.spcreator(D)
    SIJ = S[I, J]
    if issparse(SIJ):
        SIJ = SIJ.toarray()
    assert_equal(SIJ, D[I, J])
    I_bad = I + M
    J_bad = J - N
    assert_raises(IndexError, S.__getitem__, (I_bad, J))
    assert_raises(IndexError, S.__getitem__, (I, J_bad))