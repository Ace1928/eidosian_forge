import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_issue_15093(self):
    x0 = np.array([0.0, 0.5])

    def obj(x):
        x1 = x[0]
        x2 = x[1]
        return x1 ** 2 + x2 ** 2
    bounds = Bounds(np.array([0.0, 0.0]), np.array([1.0, 1.0]), keep_feasible=True)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'delta_grad == 0.0')
        result = minimize(method='trust-constr', fun=obj, x0=x0, bounds=bounds)
    assert result['success']