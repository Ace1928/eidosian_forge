import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.errors import IterationLimitError
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.sympy_tools import sympy_available
@unittest.skipUnless(differentiate_available, 'this test requires sympy')
def test_nonlinear_bound_violation(self):
    m = ConcreteModel()
    m.v1 = Var(initialize=1, domain=NonNegativeReals)
    m.c1 = Constraint(expr=m.v1 == 0)
    m.c4 = Constraint(expr=m.v1 ** 3 == -8)
    for mode in all_diff_modes:
        m.v1.set_value(1)
        calculate_variable_from_constraint(m.v1, m.c4, diff_mode=mode)
        self.assertEqual(value(m.v1), -2)