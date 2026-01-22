import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_issue_43(self):
    model = ConcreteModel()
    model.Jobs = Set(initialize=[0, 1, 2, 3])
    model.Dummy = Set(model.Jobs, within=model.Jobs, initialize=lambda m, i: range(i))
    model.Cars = Set(initialize=['a', 'b'])
    a = model.Cars * model.Dummy[1]
    self.assertEqual(len(a), 2)
    self.assertIn(('a', 0), a)
    self.assertIn(('b', 0), a)
    b = model.Dummy[2] * model.Cars
    self.assertEqual(len(b), 4)
    self.assertIn((0, 'a'), b)
    self.assertIn((0, 'b'), b)
    self.assertIn((1, 'a'), b)
    self.assertIn((1, 'b'), b)