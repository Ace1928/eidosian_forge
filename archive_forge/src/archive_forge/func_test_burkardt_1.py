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
def test_burkardt_1(self):
    exp1 = np.exp(1)
    exp2 = np.exp(2)
    A = np.array([[1, 0], [0, 2]], dtype=float)
    desired = np.array([[exp1, 0], [0, exp2]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)