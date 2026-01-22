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
def test_burkardt_2(self):
    A = np.array([[1, 3], [3, 2]], dtype=float)
    desired = np.array([[39.32280970803386, 46.16630143888575], [46.16630143888577, 54.71157685432911]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)