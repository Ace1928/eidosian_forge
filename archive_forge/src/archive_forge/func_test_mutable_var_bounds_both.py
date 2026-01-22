import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Var, Constraint, value
def test_mutable_var_bounds_both(self):
    model = AbstractModel()
    model.P = Param(initialize=4.0, mutable=True)
    model.Q = Param(initialize=2.0, mutable=True)
    model.X = Var(bounds=(model.P, model.Q))
    instance = model.create_instance()
    self.assertEqual(value(instance.X.lb), 4.0)
    self.assertEqual(value(instance.X.ub), 2.0)
    instance.P = 8.0
    instance.Q = 1.0
    self.assertEqual(value(instance.X.lb), 8.0)
    self.assertEqual(value(instance.X.ub), 1.0)