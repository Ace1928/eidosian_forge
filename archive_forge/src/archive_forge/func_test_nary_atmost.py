import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_nary_atmost(self):
    nargs = 5
    m = ConcreteModel()
    m.s = RangeSet(nargs)
    m.Y = BooleanVar(m.s)
    for truth_combination in _generate_possible_truth_inputs(nargs):
        for ntrue in range(nargs + 1):
            m.Y.set_values(dict(enumerate(truth_combination, 1)))
            correct_value = sum(truth_combination) <= ntrue
            self.assertEqual(value(atmost(ntrue, *(m.Y[i] for i in m.s))), correct_value)
            self.assertEqual(value(atmost(ntrue, m.Y)), correct_value)