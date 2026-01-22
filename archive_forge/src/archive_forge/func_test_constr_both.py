import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Var, Constraint
def test_constr_both(self):
    model = AbstractModel()
    model.A = Param(default=2.0, mutable=True)
    model.B = Param(default=1.5, mutable=True)
    model.C = Param(default=2.5, mutable=True)
    model.X = Var()

    def constr_rule(model):
        return (model.A * (model.B - model.C), model.X, model.A * (model.B + model.C))
    model.constr = Constraint(rule=constr_rule)
    instance = model.create_instance()
    self.assertEqual(instance.constr.lower(), -2.0)
    self.assertEqual(instance.constr.upper(), 8.0)