import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_logical_xor_on_indirection(self):
    m = ConcreteModel()
    m.b = BooleanVar([2, 3, 4, 5])
    m.b[4].fix(False)
    m.x = Var(domain=Integers, bounds=(3, 5))
    e = m.b[m.x].xor(m.x == 5)
    visitor = self.get_visitor()
    expr = visitor.walk_expression((e, e, 0))
    self.assertIn(id(m.x), visitor.var_map)
    self.assertIn(id(m.b[3]), visitor.var_map)
    self.assertIn(id(m.b[5]), visitor.var_map)
    x = visitor.var_map[id(m.x)]
    b3 = visitor.var_map[id(m.b[3])]
    b5 = visitor.var_map[id(m.b[5])]
    self.assertTrue(expr[1].equals(cp.equal(cp.count([cp.element([b3, False, b5], 0 + 1 * (x - 3) // 1) == True, cp.equal(x, 5)], 1), 1)))