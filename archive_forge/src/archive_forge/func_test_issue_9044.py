import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_issue_9044(self):

    def callback(x, info):
        assert_('nit' in info)
        assert_('niter' in info)
    result = minimize(lambda x: x ** 2, [0], jac=lambda x: 2 * x, hess=lambda x: 2, callback=callback, method='trust-constr')
    assert_(result.get('success'))
    assert_(result.get('nit', -1) == 1)
    assert_(result.get('niter', -1) == 1)