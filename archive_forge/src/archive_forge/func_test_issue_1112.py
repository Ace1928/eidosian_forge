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
def test_issue_1112(self):
    m = ConcreteModel()
    m.a = Set(initialize=[1, 2, 3])
    vals = list(m.a.values())
    self.assertEqual(len(vals), 1)
    self.assertIs(vals[0], m.a)
    cross = m.a.cross(m.a)
    self.assertIs(type(cross), SetProduct_OrderedSet)
    vals = list(m.a.cross(m.a).values())
    self.assertEqual(len(vals), 1)
    self.assertIsInstance(vals[0], SetProduct_OrderedSet)
    self.assertIsNot(vals[0], cross)