import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr.numeric_expr import LinearExpression, MonomialTermExpression
from pyomo.core.expr import Expr_if, inequality, LinearExpression, NPV_SumExpression
import pyomo.repn.linear as linear
from pyomo.repn.linear import LinearRepn, LinearRepnVisitor
from pyomo.repn.util import InvalidNumber
from pyomo.environ import (
def test_to_expression(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    cfg = VisitorConfig()
    visitor = LinearRepnVisitor(*cfg)
    visitor.walk_expression(m.x + m.y)
    expr = LinearRepn()
    self.assertEqual(expr.to_expression(visitor), 0)
    expr.linear[id(m.x)] = 0
    self.assertEqual(expr.to_expression(visitor), 0)
    expr.linear[id(m.x)] = 1
    assertExpressionsEqual(self, expr.to_expression(visitor), m.x)
    expr.linear[id(m.x)] = 2
    assertExpressionsEqual(self, expr.to_expression(visitor), 2 * m.x)
    expr.linear[id(m.y)] = 3
    assertExpressionsEqual(self, expr.to_expression(visitor), 2 * m.x + 3 * m.y)
    expr.multiplier = 10
    assertExpressionsEqual(self, expr.to_expression(visitor), (2 * m.x + 3 * m.y) * 10)
    expr.multiplier = 1
    expr.constant = 0
    expr.linear[id(m.x)] = 0
    expr.linear[id(m.y)] = 0
    assertExpressionsEqual(self, expr.to_expression(visitor), LinearExpression())