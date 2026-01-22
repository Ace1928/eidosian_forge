import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_double_indirection_before_constraint(self):
    m = self.get_model()
    m.i3 = IntervalVar([(1, 3), (1, 4), (1, 5)], length=4, optional=True)
    m.y = Var(domain=Integers, bounds=[1, 2])
    m.c = LogicalConstraint(expr=m.i3[1, m.x - 3].start_time.before(m.i2[m.y].end_time))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.expr, m.c, 0))
    self.assertIn(id(m.y), visitor.var_map)
    self.assertIn(id(m.i2[1]), visitor.var_map)
    self.assertIn(id(m.i2[2]), visitor.var_map)
    self.assertIn(id(m.i3[1, 3]), visitor.var_map)
    self.assertIn(id(m.i3[1, 4]), visitor.var_map)
    self.assertIn(id(m.i3[1, 5]), visitor.var_map)
    y = visitor.var_map[id(m.y)]
    x = visitor.var_map[id(m.x)]
    i21 = visitor.var_map[id(m.i2[1])]
    i22 = visitor.var_map[id(m.i2[2])]
    i33 = visitor.var_map[id(m.i3[1, 3])]
    i34 = visitor.var_map[id(m.i3[1, 4])]
    i35 = visitor.var_map[id(m.i3[1, 5])]
    self.assertTrue(expr[1].equals(cp.element([cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)], 0 + 1 * (x + -3 - 3) // 1) <= cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)))