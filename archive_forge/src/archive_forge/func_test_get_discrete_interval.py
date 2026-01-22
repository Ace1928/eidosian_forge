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
def test_get_discrete_interval(self):
    self.assertEqual(Integers.get_interval(), (None, None, 1))
    self.assertEqual(PositiveIntegers.get_interval(), (1, None, 1))
    self.assertEqual(NegativeIntegers.get_interval(), (None, -1, 1))
    self.assertEqual(Binary.get_interval(), (0, 1, 1))
    a = PositiveIntegers | NegativeIntegers
    self.assertEqual(a.get_interval(), (None, None, None))
    a = NegativeIntegers | NonNegativeIntegers
    self.assertEqual(a.get_interval(), (None, None, 1))
    a = SetOf([1, 3, 5, 6, 4, 2])
    self.assertEqual(a.get_interval(), (1, 6, 1))
    a = SetOf([1, 3, 5, 6, 2])
    self.assertEqual(a.get_interval(), (1, 6, None))
    a = SetOf([1, 3, 5, 6, 4, 2, 'a'])
    self.assertEqual(a.get_interval(), (None, None, None))
    a = SetOf([3])
    self.assertEqual(a.get_interval(), (3, 3, 0))
    a = RangeSet(ranges=(NR(0, 5, 1), NR(5, 10, 1)))
    self.assertEqual(a.get_interval(), (0, 10, 1))
    a = RangeSet(ranges=(NR(5, 10, 1), NR(0, 5, 1)))
    self.assertEqual(a.get_interval(), (0, 10, 1))
    a = RangeSet(ranges=(NR(0, 4, 1), NR(5, 10, 1)))
    self.assertEqual(a.get_interval(), (0, 10, 1))
    a = RangeSet(ranges=(NR(5, 10, 1), NR(0, 4, 1)))
    self.assertEqual(a.get_interval(), (0, 10, 1))
    a = RangeSet(ranges=(NR(0, 3, 1), NR(5, 10, 1)))
    self.assertEqual(a.get_interval(), (0, 10, None))
    a = RangeSet(ranges=(NR(5, 10, 1), NR(0, 3, 1)))
    self.assertEqual(a.get_interval(), (0, 10, None))
    a = RangeSet(ranges=(NR(0, 4, 2), NR(6, 10, 2)))
    self.assertEqual(a.get_interval(), (0, 10, 2))
    a = RangeSet(ranges=(NR(6, 10, 2), NR(0, 4, 2)))
    self.assertEqual(a.get_interval(), (0, 10, 2))
    a = RangeSet(ranges=(NR(0, 4, 2), NR(5, 10, 2)))
    self.assertEqual(a.get_interval(), (0, 9, None))
    a = RangeSet(ranges=(NR(5, 10, 2), NR(0, 4, 2)))
    self.assertEqual(a.get_interval(), (0, 9, None))
    a = RangeSet(ranges=(NR(0, 10, 2), NR(0, 10, 3)))
    self.assertEqual(a.get_interval(), (0, 10, None))
    a = RangeSet(ranges=(NR(0, 10, 3), NR(0, 10, 2)))
    self.assertEqual(a.get_interval(), (0, 10, None))
    a = RangeSet(ranges=(NR(2, 10, 2), NR(0, 12, 4)))
    self.assertEqual(a.get_interval(), (0, 12, 2))
    a = RangeSet(ranges=(NR(0, 12, 4), NR(2, 10, 2)))
    self.assertEqual(a.get_interval(), (0, 12, 2))
    a = RangeSet(ranges=(NR(0, 10, 2), NR(1, 10, 2)))
    self.assertEqual(a.get_interval(), (0, 10, None))
    a = RangeSet(ranges=(NR(0, 10, 3), NR(1, 10, 3), NR(2, 10, 3)))
    self.assertEqual(a.get_interval(), (0, 10, None))