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
def test_clone_ConstraintElement(self):
    model = ConcreteModel()
    model.x = Var()
    model.c = Constraint(expr=model.x == 1.0)
    model.junk = Suffix()
    self.assertEqual(model.junk.get(model.c), None)
    model.junk.set_value(model.c, 1.0)
    self.assertEqual(model.junk.get(model.c), 1.0)
    inst = model.clone()
    self.assertEqual(inst.junk.get(model.c), None)
    self.assertEqual(inst.junk.get(inst.c), 1.0)