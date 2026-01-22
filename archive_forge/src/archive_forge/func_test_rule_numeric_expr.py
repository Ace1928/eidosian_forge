import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_rule_numeric_expr(self):
    """Test rule option with returns a single numeric constant for the expression"""
    model = self.create_model()

    def f(model, i):
        return 1.0
    model.obj = Objective(model.A, rule=f)
    self.assertEqual(model.obj[2](), 1.0)
    self.assertEqual(value(model.obj[2]), 1.0)