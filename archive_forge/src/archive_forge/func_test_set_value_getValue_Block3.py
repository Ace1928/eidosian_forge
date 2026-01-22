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
def test_set_value_getValue_Block3(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.b = Block()
    model.B = Block([1, 2, 3])
    model.B[1].x = 1
    model.B[2].x = 2
    model.B[3].x = 3
    model.B.set_suffix_value(model.junk, 1.0)
    model.B[1].set_suffix_value(model.junk, 2.0)
    self.assertEqual(model.B.get_suffix_value(model.junk), None)
    self.assertEqual(model.B[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.B[2].get_suffix_value(model.junk), 1.0)
    self.assertEqual(model.b.get_suffix_value(model.junk), None)
    model.b.set_suffix_value(model.junk, 3.0)
    model.B[2].set_suffix_value(model.junk, 3.0)
    self.assertEqual(model.B.get_suffix_value(model.junk), None)
    self.assertEqual(model.B[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.B[2].get_suffix_value(model.junk), 3.0)
    self.assertEqual(model.b.get_suffix_value(model.junk), 3.0)
    model.B.set_suffix_value(model.junk, 1.0, expand=False)
    self.assertEqual(model.B.get_suffix_value(model.junk), 1.0)