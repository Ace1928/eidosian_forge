import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle1(self):
    model = AbstractModel()
    model.A = Set(initialize=[1, 2, 3])
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300}, mutable=True)
    model.x = Var(model.A)
    model.y = Var(model.A)
    model.obj = Objective(rule=obj_rule)
    model.constr = Constraint(model.A, rule=constr_rule)
    pickle_str = pickle.dumps(model)
    tmodel = pickle.loads(pickle_str)
    instance = tmodel.create_instance()
    expr = sum_product(instance.x, instance.B, instance.y)
    baseline = 'B[1]*x[1]*y[1] + B[2]*x[2]*y[2] + B[3]*x[3]*y[3]'
    self.assertEqual(str(expr), baseline)