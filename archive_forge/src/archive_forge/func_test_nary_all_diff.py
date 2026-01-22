import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_nary_all_diff(self):
    m = ConcreteModel()
    m.x = Var(range(4), domain=Integers, bounds=(0, 3))
    for vals in permutations(range(4)):
        self.assertTrue(value(all_different(*vals)))
        for i, v in enumerate(vals):
            m.x[i] = v
        self.assertTrue(value(all_different(m.x)))
    self.assertFalse(value(all_different(1, 1, 2, 3)))
    m.x[0] = 1
    m.x[1] = 1
    m.x[2] = 2
    m.x[3] = 3
    self.assertFalse(value(all_different(m.x)))