import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_init_concrete_indexed(self):
    model = ConcreteModel()
    model.y = Var(initialize=0.0)
    model.x = Var([1, 2, 3], initialize=1.0)
    model.ec = Expression([1, 2, 3], initialize=1.0)
    model.obj = Objective(expr=1.0 + sum_product(model.ec, index=[1, 2, 3]))
    self.assertEqual(model.obj.expr(), 4.0)
    model.ec[1].set_value(2.0)
    self.assertEqual(model.obj.expr(), 5.0)