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
def test_set_value_getValue_Set3(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.s = Set(initialize=[1, 2, 3])
    model.S = Set([1, 2, 3], initialize={1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]})
    model.S.set_suffix_value(model.junk, 1.0)
    model.S[1].set_suffix_value(model.junk, 2.0)
    self.assertEqual(model.S.get_suffix_value(model.junk), None)
    self.assertEqual(model.S[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.S[2].get_suffix_value(model.junk), 1.0)
    self.assertEqual(model.s.get_suffix_value(model.junk), None)
    model.s.set_suffix_value(model.junk, 3.0)
    model.S[2].set_suffix_value(model.junk, 3.0)
    self.assertEqual(model.S.get_suffix_value(model.junk), None)
    self.assertEqual(model.S[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.S[2].get_suffix_value(model.junk), 3.0)
    self.assertEqual(model.s.get_suffix_value(model.junk), 3.0)
    model.S.set_suffix_value(model.junk, 1.0, expand=False)
    self.assertEqual(model.S.get_suffix_value(model.junk), 1.0)