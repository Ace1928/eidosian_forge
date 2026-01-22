import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def test_pickle_ObjectiveArray(self):
    model = ConcreteModel()
    model.x = Var([1, 2, 3], dense=True)
    model.obj = Objective([1, 2, 3], rule=simple_obj_rule)
    model.junk = Suffix()
    self.assertEqual(model.junk.get(model.obj), None)
    self.assertEqual(model.junk.get(model.obj[1]), None)
    model.junk.set_value(model.obj, 1.0)
    self.assertEqual(model.junk.get(model.obj), None)
    self.assertEqual(model.junk.get(model.obj[1]), 1.0)
    inst = pickle.loads(pickle.dumps(model))
    self.assertEqual(inst.junk.get(model.obj[1]), None)
    self.assertEqual(inst.junk.get(inst.obj[1]), 1.0)