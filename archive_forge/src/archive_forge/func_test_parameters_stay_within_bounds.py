from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_parameters_stay_within_bounds(self):
    np.random.seed(1)
    bounds = Bounds(np.array([0.1]), np.array([1.0]))
    n_inputs = len(bounds.lb)
    x0 = np.array(bounds.lb + (bounds.ub - bounds.lb) * np.random.random(n_inputs))

    def f(x):
        assert (x >= bounds.lb).all()
        return np.linalg.norm(x)
    with pytest.warns(RuntimeWarning, match='x were outside bounds'):
        res = minimize(f, x0, method='SLSQP', bounds=bounds)
        assert res.success