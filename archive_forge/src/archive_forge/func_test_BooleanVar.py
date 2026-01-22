import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_BooleanVar(self):
    """
        Simple construction and value setting
        """
    m = ConcreteModel()
    m.Y1 = BooleanVar()
    m.Y2 = BooleanVar()
    self.assertIsNone(m.Y1.value)
    m.Y1.set_value(False)
    self.assertFalse(m.Y1.value)
    m.Y1.set_value(True)
    self.assertTrue(m.Y1.value)