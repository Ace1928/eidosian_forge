from pyomo.common import unittest
import pyomo.environ as pe
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.compare import (
from pyomo.common.gsl import find_GSL
def test_ranged_expression(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    e = pe.inequality(-1, m.x, 1)
    pn = convert_expression_to_prefix_notation(e)
    expected = [(RangedExpression, 3), -1, m.x, 1]
    self.assertEqual(pn, expected)