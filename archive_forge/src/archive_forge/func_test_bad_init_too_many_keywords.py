import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_bad_init_too_many_keywords(self):
    model = ConcreteModel()

    def _some_rule(model):
        return 1.0
    with self.assertRaises(ValueError):
        model.e = Expression(expr=1.0, rule=_some_rule)
    del _some_rule

    def _some_indexed_rule(model, i):
        return 1.0
    with self.assertRaises(ValueError):
        model.e = Expression([1], expr=1.0, rule=_some_indexed_rule)
    del _some_indexed_rule