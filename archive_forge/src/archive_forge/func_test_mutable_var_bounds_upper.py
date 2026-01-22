import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Var, Constraint, value
def test_mutable_var_bounds_upper(self):
    model = AbstractModel()
    model.Q = Param(initialize=2.0, mutable=True)
    model.X = Var(bounds=(model.Q, None))
    instance = model.create_instance()
    self.assertEqual(instance.X.bounds, (2.0, None))
    instance.Q = 4.0
    self.assertEqual(instance.X.bounds, (4.0, None))