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
def test_initialize_value(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var(initialize=0)
    m.c = Constraint(expr=m.x == 5)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
        self.assertEqual(value(m.x), 5)
    m.x.setlb(3)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
        self.assertEqual(value(m.x), 5)
    m.x.setlb(-10)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
        self.assertEqual(value(m.x), 5)
    m.x.setub(10)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
        self.assertEqual(value(m.x), 5)
    m.x.setlb(3)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
        self.assertEqual(value(m.x), 5)
    m.x.setlb(None)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
        self.assertEqual(value(m.x), 5)
    m.x.setub(-10)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
        self.assertEqual(value(m.x), 5)
    m.lt = Constraint(expr=m.x <= m.y)
    with self.assertRaisesRegex(ValueError, "Constraint 'lt' must be an equality constraint"):
        calculate_variable_from_constraint(m.x, m.lt)