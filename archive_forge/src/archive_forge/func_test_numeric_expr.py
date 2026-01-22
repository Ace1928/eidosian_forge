import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_numeric_expr(self):
    """Test expr option with a single numeric constant"""
    model = ConcreteModel()
    model.obj = Objective(expr=0.0)
    self.assertEqual(model.obj(), 0.0)
    self.assertEqual(value(model.obj), 0.0)
    self.assertEqual(value(model.obj._data[None]), 0.0)