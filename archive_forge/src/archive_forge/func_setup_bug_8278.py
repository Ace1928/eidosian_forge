import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
def setup_bug_8278():
    N = 2 ** 6
    h = 1 / N
    Ah1D = scipy.sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(N - 1, N - 1)) / h ** 2
    eyeN = scipy.sparse.eye(N - 1)
    A = scipy.sparse.kron(eyeN, scipy.sparse.kron(eyeN, Ah1D)) + scipy.sparse.kron(eyeN, scipy.sparse.kron(Ah1D, eyeN)) + scipy.sparse.kron(Ah1D, scipy.sparse.kron(eyeN, eyeN))
    b = np.random.rand((N - 1) ** 3)
    return (A, b)