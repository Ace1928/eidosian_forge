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
@unittest.skipUnless(numpy_available, 'Test requires numpy')
def test_nonnumeric(self):
    m = ConcreteModel()
    m.p = Param(mutable=True, initialize=numpy.array([3]), domain=Any)
    m.e = Expression()
    cfg = VisitorConfig()
    repn = LinearRepnVisitor(*cfg).walk_expression(m.p)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {})
    self.assertEqual(cfg.var_order, {})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 3)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    m.p = numpy.array([3, 4])
    cfg = VisitorConfig()
    repn = LinearRepnVisitor(*cfg).walk_expression(m.p)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {})
    self.assertEqual(cfg.var_order, {})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(str(repn.constant), 'InvalidNumber(array([3, 4]))')
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)