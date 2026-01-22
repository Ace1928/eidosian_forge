import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
def test_Expression_component(self):
    m = ConcreteModel()
    m.s = Set(initialize=['A', 'B'])
    m.x = Var(m.s, domain=NonNegativeReals)

    def y_rule(m, s):
        return m.x[s] * 2
    m.y = Expression(m.s, rule=y_rule)
    expr = 1 - m.y['A'] ** 2
    jacs = differentiate(expr, wrt_list=[m.x['A'], m.x['B']])
    self.assertEqual(str(jacs[0]), '-8.0*x[A]')
    self.assertEqual(str(jacs[1]), '0.0')
    expr = 1 - m.y['B'] ** 2
    jacs = differentiate(expr, wrt_list=[m.x['A'], m.x['B']])
    self.assertEqual(str(jacs[0]), '0.0')
    self.assertEqual(str(jacs[1]), '-8.0*x[B]')