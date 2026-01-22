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
def test_burkardt_8(self):
    exp4 = np.exp(4)
    exp16 = np.exp(16)
    A = np.array([[21, 17, 6], [-5, -1, -6], [4, 4, 16]], dtype=float)
    desired = np.array([[13 * exp16 - exp4, 13 * exp16 - 5 * exp4, 2 * exp16 - 2 * exp4], [-9 * exp16 + exp4, -9 * exp16 + 5 * exp4, -2 * exp16 + 2 * exp4], [16 * exp16, 16 * exp16, 4 * exp16]], dtype=float) * 0.25
    actual = expm(A)
    assert_allclose(actual, desired)