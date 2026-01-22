import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def test_onenorm_matrix_power_nnm():
    np.random.seed(1234)
    for n in range(1, 5):
        for p in range(5):
            M = np.random.random((n, n))
            Mp = np.linalg.matrix_power(M, p)
            observed = _onenorm_matrix_power_nnm(M, p)
            expected = np.linalg.norm(Mp, 1)
            assert_allclose(observed, expected)