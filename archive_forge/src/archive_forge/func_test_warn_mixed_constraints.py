import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from scipy.optimize import (NonlinearConstraint, LinearConstraint,
from .test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
def test_warn_mixed_constraints(self):

    def fun(x):
        return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
    cons = NonlinearConstraint(lambda x: [x[0] ** 2 - x[1], x[1] - x[2]], [1.1, 0.8], [1.1, 1.4])
    bnds = ((0, None), (0, None), (0, None))
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'delta_grad == 0.0')
        assert_warns(OptimizeWarning, minimize, fun, (2, 0, 1), method=self.method, bounds=bnds, constraints=cons)