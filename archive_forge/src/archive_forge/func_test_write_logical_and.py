import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_write_logical_and(self):
    m = self.get_model()
    m.c = LogicalConstraint(expr=m.b.land(m.b2['b']))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.expr, m.c, 0))
    self.assertIn(id(m.b), visitor.var_map)
    self.assertIn(id(m.b2['b']), visitor.var_map)
    b = visitor.var_map[id(m.b)]
    b2b = visitor.var_map[id(m.b2['b'])]
    self.assertTrue(expr[1].equals(cp.logical_and(b, b2b)))