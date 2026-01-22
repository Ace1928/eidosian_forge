from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
@unittest.skipUnless(scipy_available and numpy_available, 'scipy and/or numpy are not available')
def test_pw_linear_approx_of_paraboloid_points(self):
    m = self.make_model()
    m.pw = PiecewiseLinearFunction(points=[(0, 1), (0, 4), (0, 7), (3, 1), (3, 4), (3, 7)], function=m.g)
    self.check_pw_linear_approximation(m)