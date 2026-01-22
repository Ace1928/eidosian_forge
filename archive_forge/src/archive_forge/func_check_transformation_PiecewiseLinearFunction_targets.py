import pyomo.contrib.piecewise.tests.models as models
from pyomo.core import Var
from pyomo.core.base import TransformationFactory
from pyomo.environ import value
from pyomo.gdp import Disjunct, Disjunction
def check_transformation_PiecewiseLinearFunction_targets(test, transformation):
    m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m, targets=[m.pw_log])
    test.check_pw_log(m)
    test.assertIsNone(m.pw_paraboloid.get_transformation_var(m.paraboloid_expr))