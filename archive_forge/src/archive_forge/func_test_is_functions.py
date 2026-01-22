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
def test_is_functions(self):
    i = SetOf({1, 2, 3})
    self.assertTrue(i.isdiscrete())
    self.assertTrue(i.isfinite())
    self.assertFalse(i.isordered())
    i = SetOf([1, 2, 3])
    self.assertTrue(i.isdiscrete())
    self.assertTrue(i.isfinite())
    self.assertTrue(i.isordered())
    i = SetOf((1, 2, 3))
    self.assertTrue(i.isdiscrete())
    self.assertTrue(i.isfinite())
    self.assertTrue(i.isordered())
    i = RangeSet(3)
    self.assertTrue(i.isdiscrete())
    self.assertTrue(i.isfinite())
    self.assertTrue(i.isordered())
    self.assertIsInstance(i, _FiniteRangeSetData)
    i = RangeSet(1, 3)
    self.assertTrue(i.isdiscrete())
    self.assertTrue(i.isfinite())
    self.assertTrue(i.isordered())
    self.assertIsInstance(i, _FiniteRangeSetData)
    i = RangeSet(1, 3, 0)
    self.assertFalse(i.isdiscrete())
    self.assertFalse(i.isfinite())
    self.assertFalse(i.isordered())
    self.assertIsInstance(i, _InfiniteRangeSetData)