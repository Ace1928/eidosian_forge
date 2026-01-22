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
def test_pickle_model(self):
    model = ConcreteModel()
    model.junk = Suffix()
    self.assertEqual(model.junk.get(model), None)
    model.junk.set_value(model, 1.0)
    self.assertEqual(model.junk.get(model), 1.0)
    inst = pickle.loads(pickle.dumps(model))
    self.assertEqual(inst.junk.get(model), None)
    self.assertEqual(inst.junk.get(inst), 1.0)