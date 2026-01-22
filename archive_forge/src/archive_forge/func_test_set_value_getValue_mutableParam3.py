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
def test_set_value_getValue_mutableParam3(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.p = Param(initialize=1.0, mutable=True)
    model.P = Param([1, 2, 3], initialize=1.0, mutable=True)
    model.P.set_suffix_value(model.junk, 1.0)
    model.P[1].set_suffix_value(model.junk, 2.0)
    self.assertEqual(model.P.get_suffix_value(model.junk), None)
    self.assertEqual(model.P[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.P[2].get_suffix_value(model.junk), 1.0)
    self.assertEqual(model.p.get_suffix_value(model.junk), None)
    model.p.set_suffix_value(model.junk, 3.0)
    model.P[2].set_suffix_value(model.junk, 3.0)
    self.assertEqual(model.P.get_suffix_value(model.junk), None)
    self.assertEqual(model.P[1].get_suffix_value(model.junk), 2.0)
    self.assertEqual(model.P[2].get_suffix_value(model.junk), 3.0)
    self.assertEqual(model.p.get_suffix_value(model.junk), 3.0)
    model.P.set_suffix_value(model.junk, 1.0, expand=False)
    self.assertEqual(model.P.get_suffix_value(model.junk), 1.0)