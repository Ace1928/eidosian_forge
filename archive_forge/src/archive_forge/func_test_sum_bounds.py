import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_sum_bounds(self):
    m = self.make_model()
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(m.x + m.y)
    self.assertEqual(lb, 1)
    self.assertEqual(ub, 9)
    self.assertEqual(len(visitor.leaf_bounds), 2)
    self.assertIn(m.x, visitor.leaf_bounds)
    self.assertIn(m.y, visitor.leaf_bounds)
    self.assertEqual(visitor.leaf_bounds[m.x], (-2, 4))
    self.assertEqual(visitor.leaf_bounds[m.y], (3, 5))