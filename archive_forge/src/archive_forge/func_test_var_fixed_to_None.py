import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_var_fixed_to_None(self):
    m = self.make_model()
    m.x.fix(None)
    visitor = ExpressionBoundsVisitor(use_fixed_var_values_as_bounds=True)
    with self.assertRaisesRegex(ValueError, "Var 'x' is fixed to None. This value cannot be used to calculate bounds."):
        lb, ub = visitor.walk_expression(m.x - m.y)