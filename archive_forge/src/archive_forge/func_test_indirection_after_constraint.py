import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_indirection_after_constraint(self):
    m = self.get_model()
    m.y = Var(domain=Integers, bounds=[1, 2])
    m.c = LogicalConstraint(expr=m.i2[m.y].start_time.after(m.i.end_time, delay=-2))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.expr, m.c, 0))
    self.assertIn(id(m.y), visitor.var_map)
    self.assertIn(id(m.i2[1]), visitor.var_map)
    self.assertIn(id(m.i2[2]), visitor.var_map)
    self.assertIn(id(m.i), visitor.var_map)
    y = visitor.var_map[id(m.y)]
    i21 = visitor.var_map[id(m.i2[1])]
    i22 = visitor.var_map[id(m.i2[2])]
    i = visitor.var_map[id(m.i)]
    self.assertTrue(expr[1].equals(cp.end_of(i) + -2 <= cp.element([cp.start_of(i21), cp.start_of(i22)], 0 + 1 * (y - 1) // 1)))