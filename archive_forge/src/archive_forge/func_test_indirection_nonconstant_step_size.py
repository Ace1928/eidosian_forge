import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_indirection_nonconstant_step_size(self):
    m = ConcreteModel()

    def param_rule(m, i):
        return i + 1
    m.p = Param([1, 3, 4], initialize=param_rule)
    m.x = Var(within={1, 3, 4})
    e = m.p[m.x]
    visitor = self.get_visitor()
    with self.assertRaisesRegex(ValueError, "Variable indirection 'p\\[x\\]' is over a discrete domain without a constant step size. This is not supported."):
        expr = visitor.walk_expression((e, e, 0))