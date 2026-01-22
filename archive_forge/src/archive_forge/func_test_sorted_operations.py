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
def test_sorted_operations(self):
    I = UnindexedComponent_set
    self.assertEqual(len(I), 1)
    self.assertEqual(I.dimen, 0)
    self.assertTrue(I.isdiscrete())
    self.assertTrue(I.isfinite())
    self.assertTrue(I.isordered())
    with self.assertRaisesRegex(AttributeError, "has no attribute 'add'"):
        I.add(1)
    with self.assertRaisesRegex(AttributeError, "has no attribute 'set_value'"):
        I.set_value(1)
    with self.assertRaisesRegex(AttributeError, "has no attribute 'remove'"):
        I.remove(val[0])
    with self.assertRaisesRegex(AttributeError, "has no attribute 'discard'"):
        I.discard(val[0])
    with self.assertRaisesRegex(AttributeError, "has no attribute 'pop'"):
        I.pop()
    with self.assertRaisesRegex(AttributeError, "has no attribute 'clear'"):
        I.clear()
    with self.assertRaisesRegex(AttributeError, "has no attribute 'update'"):
        I.update()
    self.assertEqual(str(I), 'UnindexedComponent_set')
    self.assertEqual(','.join((str(_) for _ in I.ranges())), '{None}')
    self.assertIsNone(I.construct())
    val = I.data()
    self.assertIs(type(val), tuple)
    self.assertEqual(len(val), 1)
    self.assertEqual(I.ordered_data(), val)
    self.assertEqual(I.sorted_data(), val)
    self.assertEqual(I.get(val[0], 100), val[0])
    self.assertEqual(I.get(999, 100), 100)
    self.assertEqual(tuple(I), val)
    self.assertEqual(tuple(reversed(I)), val)
    self.assertEqual(tuple(I.sorted_iter()), val)
    self.assertEqual(tuple(I.ordered_iter()), val)
    self.assertEqual(I.bounds(), (None, None))
    self.assertEqual(I.get_interval(), (None, None, None))
    self.assertEqual(I.subsets(), [I])
    self.assertEqual(I.first(), val[0])
    self.assertEqual(I.last(), val[0])
    self.assertEqual(I.at(1), val[0])
    with self.assertRaisesRegex(IndexError, 'UnindexedComponent_set index out of range'):
        I.at(999)
    self.assertEqual(I.ord(val[0]), 1)
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 999 in Set UnindexedComponent_set: item not in Set'):
        I.ord(999)
    with self.assertRaisesRegex(IndexError, 'Cannot advance past the end of the Set'):
        I.next(val[0])
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 999 in Set UnindexedComponent_set: item not in Set'):
        I.next(999)
    self.assertEqual(I.nextw(val[0]), val[0])
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 999 in Set UnindexedComponent_set: item not in Set'):
        I.nextw(999)
    with self.assertRaisesRegex(IndexError, 'Cannot advance before the beginning of the Set'):
        I.prev(val[0])
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 999 in Set UnindexedComponent_set: item not in Set'):
        I.prev(999)
    self.assertEqual(I.prevw(val[0]), val[0])
    with self.assertRaisesRegex(IndexError, 'Cannot identify position of 999 in Set UnindexedComponent_set: item not in Set'):
        I.prevw(999)