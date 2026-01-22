import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_count_if(self):
    nargs = 3
    m = ConcreteModel()
    m.s = RangeSet(nargs)
    m.Y = BooleanVar(m.s)
    m.x = Var(domain=Integers, bounds=(0, 3))
    for truth_combination in _generate_possible_truth_inputs(nargs):
        for ntrue in range(nargs + 1):
            m.Y.set_values(dict(enumerate(truth_combination, 1)))
            correct_value = sum(truth_combination)
            self.assertEqual(value(count_if(*(m.Y[i] for i in m.s))), correct_value)
            self.assertEqual(value(count_if(m.Y)), correct_value)
    m.x = 2
    self.assertEqual(value(count_if([m.Y[i] for i in m.s] + [m.x == 3])), correct_value)
    m.x = 3
    self.assertEqual(value(count_if([m.Y[i] for i in m.s] + [m.x == 3])), correct_value + 1)