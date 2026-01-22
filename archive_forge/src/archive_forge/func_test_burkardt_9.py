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
def test_burkardt_9(self):
    A = np.array([[1, 2, 2, 2], [3, 1, 1, 2], [3, 2, 1, 2], [3, 3, 3, 1]], dtype=float)
    desired = np.array([[740.7038, 610.85, 542.2743, 549.1753], [731.251, 603.5524, 535.0884, 542.2743], [823.763, 679.4257, 603.5524, 610.85], [998.4355, 823.763, 731.251, 740.7038]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)