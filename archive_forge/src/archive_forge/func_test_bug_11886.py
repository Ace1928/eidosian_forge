import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_bug_11886():

    def opt(x):
        return x[0] ** 2 + x[1] ** 2
    with np.testing.suppress_warnings() as sup:
        sup.filter(PendingDeprecationWarning)
        A = np.matrix(np.diag([1, 1]))
    lin_cons = LinearConstraint(A, -1, np.inf)
    minimize(opt, 2 * [1], constraints=lin_cons)