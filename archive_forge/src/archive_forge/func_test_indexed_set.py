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
def test_indexed_set(self):
    m = ConcreteModel()
    m.I = Set([1, 2, 3], ordered=False)
    self.assertEqual(len(m.I), 0)
    self.assertEqual(m.I.data(), {})
    m.I[1]
    self.assertEqual(len(m.I), 1)
    self.assertEqual(m.I[1], [])
    self.assertEqual(m.I.data(), {1: ()})
    self.assertEqual(m.I[2], [])
    self.assertEqual(len(m.I), 2)
    self.assertEqual(m.I.data(), {1: (), 2: ()})
    m.I[1].add(1)
    m.I[2].add(2)
    m.I[3].add(4)
    self.assertEqual(len(m.I), 3)
    self.assertEqual(list(m.I[1]), [1])
    self.assertEqual(list(m.I[2]), [2])
    self.assertEqual(list(m.I[3]), [4])
    self.assertIsNot(m.I[1], m.I[2])
    self.assertIsNot(m.I[1], m.I[3])
    self.assertIsNot(m.I[2], m.I[3])
    self.assertFalse(m.I[1].isordered())
    self.assertFalse(m.I[2].isordered())
    self.assertFalse(m.I[3].isordered())
    self.assertIs(type(m.I[1]), _FiniteSetData)
    self.assertIs(type(m.I[2]), _FiniteSetData)
    self.assertIs(type(m.I[3]), _FiniteSetData)
    self.assertEqual(m.I.data(), {1: (1,), 2: (2,), 3: (4,)})
    m = ConcreteModel()
    m.I = Set([1, 2, 3], initialize=(4, 2, 5))
    self.assertEqual(len(m.I), 3)
    self.assertEqual(list(m.I[1]), [4, 2, 5])
    self.assertEqual(list(m.I[2]), [4, 2, 5])
    self.assertEqual(list(m.I[3]), [4, 2, 5])
    self.assertIsNot(m.I[1], m.I[2])
    self.assertIsNot(m.I[1], m.I[3])
    self.assertIsNot(m.I[2], m.I[3])
    self.assertTrue(m.I[1].isordered())
    self.assertTrue(m.I[2].isordered())
    self.assertTrue(m.I[3].isordered())
    self.assertIs(type(m.I[1]), _InsertionOrderSetData)
    self.assertIs(type(m.I[2]), _InsertionOrderSetData)
    self.assertIs(type(m.I[3]), _InsertionOrderSetData)
    self.assertEqual(m.I.data(), {1: (4, 2, 5), 2: (4, 2, 5), 3: (4, 2, 5)})
    m = ConcreteModel()
    m.I = Set([1, 2, 3], initialize=(4, 2, 5), ordered=Set.SortedOrder)
    self.assertEqual(len(m.I), 3)
    self.assertEqual(list(m.I[1]), [2, 4, 5])
    self.assertEqual(list(m.I[2]), [2, 4, 5])
    self.assertEqual(list(m.I[3]), [2, 4, 5])
    self.assertIsNot(m.I[1], m.I[2])
    self.assertIsNot(m.I[1], m.I[3])
    self.assertIsNot(m.I[2], m.I[3])
    self.assertTrue(m.I[1].isordered())
    self.assertTrue(m.I[2].isordered())
    self.assertTrue(m.I[3].isordered())
    self.assertIs(type(m.I[1]), _SortedSetData)
    self.assertIs(type(m.I[2]), _SortedSetData)
    self.assertIs(type(m.I[3]), _SortedSetData)
    self.assertEqual(m.I.data(), {1: (2, 4, 5), 2: (2, 4, 5), 3: (2, 4, 5)})
    m = ConcreteModel()
    m.I = Set([1, 2, 3], ordered=True)
    self.assertEqual(len(m.I), 0)
    m.I[1] = [1, 2, 3]
    m.I[2,] = [4, 5, 6]
    self.assertEqual(sorted(m.I._data.keys()), [1, 2])
    self.assertEqual(list(m.I[1]), [1, 2, 3])
    self.assertEqual(list(m.I[2]), [4, 5, 6])
    self.assertEqual(m.I.data(), {1: (1, 2, 3), 2: (4, 5, 6)})