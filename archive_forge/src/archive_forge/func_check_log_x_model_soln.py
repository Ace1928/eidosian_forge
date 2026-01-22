import pyomo.contrib.piecewise.tests.models as models
from pyomo.core import Var
from pyomo.core.base import TransformationFactory
from pyomo.environ import value
from pyomo.gdp import Disjunct, Disjunction
def check_log_x_model_soln(test, m):
    test.assertAlmostEqual(value(m.x), 4)
    test.assertAlmostEqual(value(m.x1), 1)
    test.assertAlmostEqual(value(m.x2), 1)
    test.assertAlmostEqual(value(m.obj), m.f2(4))