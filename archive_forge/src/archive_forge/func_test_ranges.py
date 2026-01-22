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
def test_ranges(self):
    i_data = [1, 3, 2, 0]
    i = SetOf(i_data)
    r = list(i.ranges())
    self.assertEqual(len(r), 4)
    for idx, x in enumerate(r):
        self.assertIsInstance(x, NR)
        self.assertTrue(x.isfinite())
        self.assertEqual(x.start, i[idx + 1])
        self.assertEqual(x.end, i[idx + 1])
        self.assertEqual(x.step, 0)
    try:
        self.assertIn(int, native_types)
        self.assertIn(int, native_numeric_types)
        native_types.remove(int)
        native_numeric_types.remove(int)
        r = list(i.ranges())
        self.assertEqual(len(r), 4)
        for idx, x in enumerate(r):
            self.assertIsInstance(x, NR)
            self.assertTrue(x.isfinite())
            self.assertEqual(x.start, i[idx + 1])
            self.assertEqual(x.end, i[idx + 1])
            self.assertEqual(x.step, 0)
        self.assertIn(int, native_types)
        self.assertIn(int, native_numeric_types)
    finally:
        native_types.add(int)
        native_numeric_types.add(int)
    i_data.append('abc')
    try:
        self.assertIn(str, native_types)
        self.assertNotIn(str, native_numeric_types)
        native_types.remove(str)
        r = list(i.ranges())
        self.assertEqual(len(r), 5)
        self.assertNotIn(str, native_types)
        self.assertNotIn(str, native_numeric_types)
        for idx, x in enumerate(r[:-1]):
            self.assertIsInstance(x, NR)
            self.assertTrue(x.isfinite())
            self.assertEqual(x.start, i[idx + 1])
            self.assertEqual(x.end, i[idx + 1])
            self.assertEqual(x.step, 0)
        self.assertIs(type(r[-1]), NNR)
    finally:
        native_types.add(str)