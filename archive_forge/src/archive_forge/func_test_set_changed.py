import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_set_changed(self):
    model = ConcreteModel()
    model.t = ContinuousSet(initialize=[1, 2, 3])
    self.assertFalse(model.t._changed)
    model.t.set_changed(True)
    self.assertTrue(model.t._changed)
    model.t.set_changed(False)
    self.assertFalse(model.t._changed)
    with self.assertRaises(ValueError):
        model.t.set_changed(3)