import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_leaf_bounds_cached(self):
    m = self.make_model()
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(m.x - m.y)
    self.assertEqual(lb, -7)
    self.assertEqual(ub, 1)
    self.assertIn(m.x, visitor.leaf_bounds)
    self.assertEqual(visitor.leaf_bounds[m.x], m.x.bounds)
    self.assertIn(m.y, visitor.leaf_bounds)
    self.assertEqual(visitor.leaf_bounds[m.y], m.y.bounds)
    lb, ub = visitor.walk_expression(m.x ** 2 + 3)
    self.assertEqual(lb, 3)
    self.assertEqual(ub, 19)