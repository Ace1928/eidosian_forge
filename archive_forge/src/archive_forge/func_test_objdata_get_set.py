import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_objdata_get_set(self):
    model = ConcreteModel()
    model.o = Objective([1], rule=lambda m, i: 1)
    self.assertEqual(len(model.o), 1)
    self.assertEqual(model.o[1].expr, 1)
    model.o[1].expr = 2
    self.assertEqual(model.o[1].expr, 2)
    model.o[1].expr += 2
    self.assertEqual(model.o[1].expr, 4)