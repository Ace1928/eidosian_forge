import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_equal_all_bounds(self):
    prob = Rosenbrock()
    bounds = Bounds([4.0, 5.0], [4.0, 5.0])
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'Initial guess is not within the specified bounds')
        result = minimize(prob.fun, [-10, 8], method='Nelder-Mead', bounds=bounds)
        assert np.allclose(result.x, [4.0, 5.0])