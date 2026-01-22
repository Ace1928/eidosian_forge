import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_expr2_option(self):
    """Test expr option"""
    model = ConcreteModel()
    model.x = Var(initialize=2)
    model.obj = Objective(expr=model.x)
    self.assertEqual(model.obj(), 2)
    self.assertEqual(value(model.obj), 2)
    self.assertEqual(value(model.obj._data[None]), 2)