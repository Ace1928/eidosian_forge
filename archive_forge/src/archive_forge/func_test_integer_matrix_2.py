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
def test_integer_matrix_2(self):
    Q = np.array([[-500, 500, 0, 0], [0, -550, 360, 190], [0, 630, -630, 0], [0, 0, 0, 0]], dtype=np.int16)
    assert_allclose(expm(Q), expm(1.0 * Q))
    Q = csc_matrix(Q)
    assert_allclose(expm(Q).A, expm(1.0 * Q).A)