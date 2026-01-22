import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_nonindexed_construct_rule(self):
    model = ConcreteModel()

    def _some_rule(model):
        return 1.0
    model.e = Expression(rule=_some_rule)
    self.assertEqual(value(model.e), 1.0)
    model.del_component(model.e)
    del _some_rule

    def _some_rule(model):
        return Expression.Skip
    model.e = Expression(rule=_some_rule)
    self.assertEqual(len(model.e), 0)