import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_asin_bounds(self):
    m = self.make_model()
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(asin(m.z))
    self.assertAlmostEqual(lb, asin(0.5))
    self.assertAlmostEqual(ub, asin(0.75))