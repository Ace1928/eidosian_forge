import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_rule_option3(self):
    """Test rule option"""
    model = self.create_model()
    model.y = Var(initialize=2)

    def f(model):
        yield model.y
        yield (2 * model.y)
        yield (2 * model.y)
        yield ObjectiveList.End
    model.c = ObjectiveList(rule=f)
    self.assertEqual(len(model.c), 3)
    self.assertEqual(model.c[1](), 2)
    model.d = ObjectiveList(rule=f(model))
    self.assertEqual(len(model.d), 3)
    self.assertEqual(model.d[1](), 2)