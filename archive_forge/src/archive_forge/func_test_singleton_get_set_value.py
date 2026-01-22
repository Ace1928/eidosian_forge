import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_singleton_get_set_value(self):
    model = ConcreteModel()
    model.o = Objective(expr=1)
    self.assertEqual(len(model.o), 1)
    self.assertEqual(model.o.expr, 1)
    model.o.expr = 2
    self.assertEqual(model.o.expr, 2)
    model.o.expr += 2
    self.assertEqual(model.o.expr, 4)