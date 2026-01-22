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
def test_set_value_getValue_Block1(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.b = Block()
    model.B = Block([1, 2, 3])
    model.B[1].x = 1
    model.B[2].x = 2
    model.B[3].x = 3
    model.junk.set_value(model.B, 1.0)
    model.junk.set_value(model.B[1], 2.0)
    self.assertEqual(model.junk.get(model.B), None)
    self.assertEqual(model.junk.get(model.B[1]), 2.0)
    self.assertEqual(model.junk.get(model.B[2]), 1.0)
    self.assertEqual(model.junk.get(model.b), None)
    model.junk.set_value(model.b, 3.0)
    model.junk.set_value(model.B[2], 3.0)
    self.assertEqual(model.junk.get(model.B), None)
    self.assertEqual(model.junk.get(model.B[1]), 2.0)
    self.assertEqual(model.junk.get(model.B[2]), 3.0)
    self.assertEqual(model.junk.get(model.b), 3.0)
    model.junk.set_value(model.B, 1.0, expand=False)
    self.assertEqual(model.junk.get(model.B), 1.0)