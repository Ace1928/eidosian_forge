import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
def test_constraint_dictionary_2(self):

    def fun(x):
        return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    cons = {'type': 'eq', 'fun': lambda x, p1, p2: p1 * x[0] - p2 * x[1], 'args': (1, 1.1), 'jac': lambda x, p1, p2: np.array([[p1, -p2]])}
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'delta_grad == 0.0')
        res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
    assert_allclose(res.x, [1.7918552, 1.62895927])
    assert_allclose(res.fun, 1.3857466063348418)