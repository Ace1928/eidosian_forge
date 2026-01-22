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
def test_burkardt_3(self):
    exp1 = np.exp(1)
    exp39 = np.exp(39)
    A = np.array([[0, 1], [-39, -40]], dtype=float)
    desired = np.array([[39 / (38 * exp1) - 1 / (38 * exp39), -np.expm1(-38) / (38 * exp1)], [39 * np.expm1(-38) / (38 * exp1), -1 / (38 * exp1) + 39 / (38 * exp39)]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)