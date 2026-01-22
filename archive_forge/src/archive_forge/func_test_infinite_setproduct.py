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
def test_infinite_setproduct(self):
    x = PositiveIntegers * SetOf([2, 3, 5, 7])
    self.assertFalse(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertIn((1, 2), x)
    self.assertNotIn((0, 2), x)
    self.assertNotIn((1, 1), x)
    self.assertNotIn(('a', 2), x)
    self.assertNotIn((2, 'a'), x)
    x = SetOf([2, 3, 5, 7]) * PositiveIntegers
    self.assertFalse(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertIn((3, 2), x)
    self.assertNotIn((1, 2), x)
    self.assertNotIn((2, 0), x)
    self.assertNotIn(('a', 2), x)
    self.assertNotIn((2, 'a'), x)
    x = PositiveIntegers * PositiveIntegers
    self.assertFalse(x.isfinite())
    self.assertFalse(x.isordered())
    self.assertIn((3, 2), x)
    self.assertNotIn((0, 2), x)
    self.assertNotIn((2, 0), x)
    self.assertNotIn(('a', 2), x)
    self.assertNotIn((2, 'a'), x)