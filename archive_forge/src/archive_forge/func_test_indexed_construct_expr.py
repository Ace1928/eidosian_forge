import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_indexed_construct_expr(self):
    model = ConcreteModel()
    model.Index = Set(initialize=[1, 2, 3])
    model.E = Expression(model.Index, expr=Expression.Skip)
    self.assertEqual(len(model.E), 0)
    model.E = Expression(model.Index)
    self.assertEqual(model.E.extract_values(), {1: None, 2: None, 3: None})
    model.del_component(model.E)
    model.E = Expression(model.Index, expr=1.0)
    self.assertEqual(model.E.extract_values(), {1: 1.0, 2: 1.0, 3: 1.0})
    model.del_component(model.E)
    model.E = Expression(model.Index, expr={1: Expression.Skip, 2: Expression.Skip, 3: 1.0})
    self.assertEqual(model.E.extract_values(), {3: 1.0})