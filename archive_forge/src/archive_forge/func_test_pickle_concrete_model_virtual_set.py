import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle_concrete_model_virtual_set(self):
    model = ConcreteModel()
    model._a = Set(initialize=[1, 2, 3])
    model.A = model._a * model._a
    str = pickle.dumps(model)
    tmodel = pickle.loads(str)
    self.verifyModel(model, tmodel)