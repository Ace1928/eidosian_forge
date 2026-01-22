import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_nary_or(self):
    nargs = 3
    m = ConcreteModel()
    m.s = RangeSet(nargs)
    m.Y = BooleanVar(m.s)
    op_static = lor(*(m.Y[i] for i in m.s))
    op_class = BooleanConstant(False)
    for y in m.Y.values():
        op_class = op_class.lor(y)
    for truth_combination in _generate_possible_truth_inputs(nargs):
        m.Y.set_values(dict(enumerate(truth_combination, 1)))
        correct_value = any(truth_combination)
        self.assertEqual(value(op_static), correct_value)
        self.assertEqual(value(op_class), correct_value)