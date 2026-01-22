import pyomo.contrib.piecewise.tests.models as models
from pyomo.core import Var
from pyomo.core.base import TransformationFactory
from pyomo.environ import value
from pyomo.gdp import Disjunct, Disjunction
def check_descend_into_expressions_constraint_target(test, transformation):
    m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m, descend_into_expressions=True, targets=[m.indexed_c])
    test.check_pw_paraboloid(m)
    test.assertIsNone(m.pw_log.get_transformation_var(m.log_expr))