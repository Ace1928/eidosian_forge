import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_write_absolute_value_expression(self):
    m = self.get_model()
    m.a.domain = NegativeIntegers
    m.c = Constraint(expr=abs(m.a[1]) + 1 <= 3)
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.body, m.c, 0))
    self.assertIn(id(m.a[1]), visitor.var_map)
    a1 = visitor.var_map[id(m.a[1])]
    self.assertTrue(expr[1].equals(cp.abs(a1) + 1))