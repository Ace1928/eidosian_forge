import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_node_types(self):
    m = ConcreteModel()
    m.Y1 = BooleanVar()
    m.Y2 = BooleanVar()
    m.Y3 = BooleanVar()
    m.int1 = Var(domain=Integers)
    m.int2 = Var(domain=Integers)
    m.int3 = Var(domain=Integers)
    self.assertFalse(m.Y1.is_expression_type())
    self.assertTrue(lnot(m.Y1).is_expression_type())
    self.assertTrue(equivalent(m.Y1, m.Y2).is_expression_type())
    self.assertTrue(atmost(1, [m.Y1, m.Y2, m.Y3]).is_expression_type())
    self.assertTrue(all_different(m.int1, m.int2, m.int3).is_expression_type())
    self.assertTrue(count_if(m.Y1, m.Y2, m.Y3).is_expression_type())