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
def test_multiple_insertion(self):
    m = ConcreteModel()
    m.I = Set(ordered=True, initialize=[1])
    self.assertEqual(m.I.add(3, 2, 4), 3)
    self.assertEqual(tuple(m.I.data()), (1, 3, 2, 4))
    self.assertEqual(m.I.add(1, 5, 4), 1)
    self.assertEqual(tuple(m.I.data()), (1, 3, 2, 4, 5))