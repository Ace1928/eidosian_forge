import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
@pytest.mark.xfail(reason='Failing on Azure Linux and macOS builds, see gh-13846')
def test_outside_bounds_warning(self):
    prob = Rosenbrock()
    message = 'Initial guess is not within the specified bounds'
    with pytest.warns(UserWarning, match=message):
        bounds = Bounds([-np.inf, 1.0], [4.0, 5.0])
        minimize(prob.fun, [-10, 8], method='Nelder-Mead', bounds=bounds)