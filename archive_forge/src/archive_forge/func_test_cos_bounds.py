import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_cos_bounds(self):
    m = self.make_model()
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(cos(m.y))
    self.assertAlmostEqual(lb, -1)
    self.assertAlmostEqual(ub, cos(5))