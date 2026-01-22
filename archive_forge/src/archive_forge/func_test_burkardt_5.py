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
def test_burkardt_5(self):
    A = np.array([[0, 6, 0, 0], [0, 0, 6, 0], [0, 0, 0, 6], [0, 0, 0, 0]], dtype=float)
    desired = np.array([[1, 6, 18, 36], [0, 1, 6, 18], [0, 0, 1, 6], [0, 0, 0, 1]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)