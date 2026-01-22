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
def test_set_value_getValue_Constraint2(self):
    model = ConcreteModel()
    model.junk = Suffix()
    model.x = Var()
    model.X = Var([1, 2, 3])
    model.c = Constraint(expr=model.x >= 1)
    model.C = Constraint([1, 2, 3], rule=lambda model, i: model.X[i] >= 1)
    model.C.set_suffix_value('junk', 1.0)
    model.C[1].set_suffix_value('junk', 2.0)
    self.assertEqual(model.C.get_suffix_value('junk'), None)
    self.assertEqual(model.C[1].get_suffix_value('junk'), 2.0)
    self.assertEqual(model.C[2].get_suffix_value('junk'), 1.0)
    self.assertEqual(model.c.get_suffix_value('junk'), None)
    model.c.set_suffix_value('junk', 3.0)
    model.C[2].set_suffix_value('junk', 3.0)
    self.assertEqual(model.C.get_suffix_value('junk'), None)
    self.assertEqual(model.C[1].get_suffix_value('junk'), 2.0)
    self.assertEqual(model.C[2].get_suffix_value('junk'), 3.0)
    self.assertEqual(model.c.get_suffix_value('junk'), 3.0)
    model.C.set_suffix_value('junk', 1.0, expand=False)
    self.assertEqual(model.C.get_suffix_value('junk'), 1.0)