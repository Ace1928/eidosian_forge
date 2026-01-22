import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_binary_implies(self):
    m = ConcreteModel()
    m.Y1 = BooleanVar()
    m.Y2 = BooleanVar()
    op_static = implies(m.Y1, m.Y2)
    op_class = m.Y1.implies(m.Y2)
    for truth_combination in _generate_possible_truth_inputs(2):
        m.Y1.value, m.Y2.value = (truth_combination[0], truth_combination[1])
        correct_value = not truth_combination[0] or truth_combination[1]
        self.assertEqual(value(op_static), correct_value)
        self.assertEqual(value(op_class), correct_value)
        nnf = lnot(m.Y1).lor(m.Y2)
        self.assertEqual(value(op_static), value(nnf))