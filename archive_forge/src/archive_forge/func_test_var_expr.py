import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_var_expr(self):
    """Test expr option with a single var"""
    model = ConcreteModel()
    model.x = Var(initialize=1.0)
    model.obj = Objective(expr=model.x)
    self.assertEqual(model.obj(), 1.0)
    self.assertEqual(value(model.obj), 1.0)
    self.assertEqual(value(model.obj._data[None]), 1.0)