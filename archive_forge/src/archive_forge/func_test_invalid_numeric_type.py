import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_invalid_numeric_type(self):
    m = self.make_model()
    m.p = Param(initialize=True, mutable=True, domain=Any)
    visitor = ExpressionBoundsVisitor()
    with self.assertRaisesRegex(ValueError, 'True \\(bool\\) is not a valid numeric type. Cannot compute bounds on expression.'):
        lb, ub = visitor.walk_expression(m.p + m.y)
    m.p.set_value(None)
    with self.assertRaisesRegex(ValueError, 'None \\(NoneType\\) is not a valid numeric type. Cannot compute bounds on expression.'):
        lb, ub = visitor.walk_expression(m.p + m.y)