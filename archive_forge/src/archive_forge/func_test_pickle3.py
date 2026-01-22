import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle3(self):

    def rule1(model):
        return (1, model.x + model.y[1], 2)

    def rule2(model, i):
        return (1, model.x + model.y[1] + i, 2)
    model = AbstractModel()
    model.a = Set(initialize=[1, 2, 3])
    model.A = Param(initialize=1, mutable=True)
    model.B = Param(model.a, mutable=True)
    model.x = Var(initialize=1, within=Reals)
    model.y = Var(model.a, initialize=1, within=Reals)
    model.obj = Objective(rule=lambda model: model.x + model.y[1])
    model.obj2 = Objective(model.a, rule=lambda model, i: i + model.x + model.y[1])
    model.con = Constraint(rule=rule1)
    model.con2 = Constraint(model.a, rule=rule2)
    instance = model.create_instance()
    if is_pypy:
        str_ = pickle.dumps(instance)
        tmp_ = pickle.loads(str_)
    else:
        with self.assertRaises((pickle.PicklingError, TypeError, AttributeError)):
            pickle.dumps(instance)