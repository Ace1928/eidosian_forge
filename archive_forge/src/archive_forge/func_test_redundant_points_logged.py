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
def test_redundant_points_logged(self):
    m = self.make_model()
    m.points.append((-2, 0, 1))
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.contrib.piecewise.piecewise_linear_function', level=logging.INFO):
        m.approx = PiecewiseLinearFunction(points=m.points, function=m.f)
    self.assertIn('The Delaunay triangulation dropped the point with index 27 from the triangulation', out.getvalue())