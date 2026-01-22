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
def test_exp_sinch_overflow(self):
    L = np.array([[1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, -0.5, -0.5], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    E0 = expm(-L)
    E1 = expm(-2 ** 11 * L)
    E2 = E0
    for j in range(11):
        E2 = E2 @ E2
    assert_allclose(E1, E2)