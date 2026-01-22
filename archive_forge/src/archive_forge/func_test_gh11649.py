import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
@pytest.mark.xfail(reason='Known bug in trust-constr; see gh-11649.', strict=True)
def test_gh11649():
    bnds = Bounds(lb=[-1, -1], ub=[1, 1], keep_feasible=True)

    def assert_inbounds(x):
        assert np.all(x >= bnds.lb)
        assert np.all(x <= bnds.ub)

    def obj(x):
        assert_inbounds(x)
        return np.exp(x[0]) * (4 * x[0] ** 2 + 2 * x[1] ** 2 + 4 * x[0] * x[1] + 2 * x[1] + 1)

    def nce(x):
        assert_inbounds(x)
        return x[0] ** 2 + x[1]

    def nci(x):
        assert_inbounds(x)
        return x[0] * x[1]
    x0 = np.array((0.99, -0.99))
    nlcs = [NonlinearConstraint(nci, -10, np.inf), NonlinearConstraint(nce, 1, 1)]
    res = minimize(fun=obj, x0=x0, method='trust-constr', bounds=bnds, constraints=nlcs)
    assert res.success
    assert_inbounds(res.x)
    assert nlcs[0].lb < nlcs[0].fun(res.x) < nlcs[0].ub
    assert_allclose(nce(res.x), nlcs[1].ub)
    ref = minimize(fun=obj, x0=x0, method='slsqp', bounds=bnds, constraints=nlcs)
    assert_allclose(res.fun, ref.fun)