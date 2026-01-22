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
def test_matrix_power():
    np.random.seed(1234)
    row, col = np.random.randint(0, 4, size=(2, 6))
    data = np.random.random(size=(6,))
    Amat = csc_matrix((data, (row, col)), shape=(4, 4))
    A = csc_array((data, (row, col)), shape=(4, 4))
    Adense = A.toarray()
    for power in (2, 5, 6):
        Apow = matrix_power(A, power).toarray()
        Amat_pow = (Amat ** power).toarray()
        Adense_pow = np.linalg.matrix_power(Adense, power)
        assert_allclose(Apow, Adense_pow)
        assert_allclose(Apow, Amat_pow)