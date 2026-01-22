import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
def test_val2(self):
    m = ConcreteModel()
    m.v = Var(initialize=2)
    e = 1 < m.v
    e = e <= 2
    self.assertEqual(value(e), True)
    e = 1 <= m.v
    e = e < 2
    self.assertEqual(value(e), False)