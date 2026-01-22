import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_exprdata_get_set_value(self):
    model = ConcreteModel()
    model.e = Expression([1])
    self.assertEqual(len(model.e), 1)
    self.assertEqual(model.e[1].expr, None)
    model.e.add(1, 1)
    model.e[1].expr = 1
    self.assertEqual(model.e[1].expr, 1)
    model.e[1].expr += 2
    self.assertEqual(model.e[1].expr, 3)