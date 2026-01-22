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
def test_pickle_VarArray(self):
    model = ConcreteModel()
    model.x = Var([1, 2, 3], dense=True)
    model.junk = Suffix()
    self.assertEqual(model.junk.get(model.x), None)
    self.assertEqual(model.junk.get(model.x[1]), None)
    model.junk.set_value(model.x, 1.0)
    self.assertEqual(model.junk.get(model.x), None)
    self.assertEqual(model.junk.get(model.x[1]), 1.0)
    inst = pickle.loads(pickle.dumps(model))
    self.assertEqual(inst.junk.get(model.x[1]), None)
    self.assertEqual(inst.junk.get(inst.x[1]), 1.0)