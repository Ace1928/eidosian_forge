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
def test_get_continuous_interval(self):
    self.assertEqual(Reals.get_interval(), (None, None, 0))
    self.assertEqual(PositiveReals.get_interval(), (0, None, 0))
    self.assertEqual(NonNegativeReals.get_interval(), (0, None, 0))
    self.assertEqual(NonPositiveReals.get_interval(), (None, 0, 0))
    self.assertEqual(NegativeReals.get_interval(), (None, 0, 0))
    a = NonNegativeReals | NonPositiveReals
    self.assertEqual(a.get_interval(), (None, None, 0))
    a = NonPositiveReals | NonNegativeReals
    self.assertEqual(a.get_interval(), (None, None, 0))
    a = NegativeReals | PositiveReals
    self.assertEqual(a.get_interval(), (None, None, None))
    a = NegativeReals | PositiveReals | [0]
    self.assertEqual(a.get_interval(), (None, None, 0))
    a = NegativeReals | PositiveReals | RangeSet(0, 5)
    self.assertEqual(a.get_interval(), (None, None, 0))
    a = NegativeReals | RangeSet(-3, 3)
    self.assertEqual(a.get_interval(), (None, 3, None))
    a = NegativeReals | Binary
    self.assertEqual(a.get_interval(), (None, 1, None))
    a = PositiveReals | Binary
    self.assertEqual(a.get_interval(), (0, None, 0))
    a = RangeSet(1, 10, 0) | RangeSet(5, 15, 0)
    self.assertEqual(a.get_interval(), (1, 15, 0))
    a = RangeSet(5, 15, 0) | RangeSet(1, 10, 0)
    self.assertEqual(a.get_interval(), (1, 15, 0))
    a = RangeSet(5, 15, 0) | RangeSet(1, 4, 0)
    self.assertEqual(a.get_interval(), (1, 15, None))
    a = RangeSet(1, 4, 0) | RangeSet(5, 15, 0)
    self.assertEqual(a.get_interval(), (1, 15, None))
    a = NegativeReals | Any
    self.assertEqual(a.get_interval(), (None, None, None))
    a = Any | NegativeReals
    self.assertEqual(a.get_interval(), (None, None, None))
    a = SetOf('abc') | NegativeReals
    self.assertEqual(a.get_interval(), (None, None, None))
    a = NegativeReals | SetOf('abc')
    self.assertEqual(a.get_interval(), (None, None, None))