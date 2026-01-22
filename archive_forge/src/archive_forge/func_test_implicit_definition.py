import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_implicit_definition(self):
    model = ConcreteModel()
    model.idx = Set(initialize=[1, 2, 3])
    model.E = Expression(model.idx)
    self.assertEqual(len(model.E), 3)
    expr = model.E[1]
    self.assertIs(type(expr), _GeneralExpressionData)
    model.E[1] = None
    self.assertIs(expr, model.E[1])
    self.assertIs(type(expr), _GeneralExpressionData)
    self.assertIs(expr.expr, None)
    model.E[1] = 5
    self.assertIs(expr, model.E[1])
    self.assertEqual(model.E.extract_values(), {1: 5, 2: None, 3: None})
    model.E[2] = 6
    self.assertIsNot(expr, model.E[2])
    self.assertEqual(model.E.extract_values(), {1: 5, 2: 6, 3: None})