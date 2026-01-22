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
def test_unordered_insertion_deletion(self):

    def _verify(_s, _l):
        self.assertFalse(_s.isordered())
        self.assertTrue(_s.isfinite())
        self.assertEqual(sorted(_s), _l)
        self.assertEqual(list(_s), list(reversed(list(reversed(_s)))))
        for v in _l:
            self.assertIn(v, _s)
        if _l:
            _max = max(_l)
            _min = min(_l)
        else:
            _max = 0
            _min = 0
        self.assertNotIn(_max + 1, _s)
        self.assertNotIn(_min - 1, _s)
    m = ConcreteModel()
    m.I = Set(ordered=False)
    _verify(m.I, [])
    m.I.add(1)
    _verify(m.I, [1])
    m.I.add(3)
    _verify(m.I, [1, 3])
    m.I.add(2)
    _verify(m.I, [1, 2, 3])
    m.I.add(4)
    _verify(m.I, [1, 2, 3, 4])
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m.I.add(3)
    self.assertEqual(output.getvalue(), 'Element 3 already exists in Set I; no action taken\n')
    _verify(m.I, [1, 2, 3, 4])
    m.I.remove(3)
    _verify(m.I, [1, 2, 4])
    with self.assertRaisesRegex(KeyError, '^3$'):
        m.I.remove(3)
    _verify(m.I, [1, 2, 4])
    m.I.add(3)
    _verify(m.I, [1, 2, 3, 4])
    m.I.discard(3)
    _verify(m.I, [1, 2, 4])
    m.I.discard(3)
    _verify(m.I, [1, 2, 4])
    m.I.clear()
    _verify(m.I, [])
    m.I.add(6)
    m.I.add(5)
    _verify(m.I, [5, 6])
    tmp = set()
    tmp.add(m.I.pop())
    tmp.add(m.I.pop())
    _verify(m.I, [])
    self.assertEqual(tmp, {5, 6})
    with self.assertRaisesRegex(KeyError, 'pop from an empty set'):
        m.I.pop()
    m.I.update([5])
    _verify(m.I, [5])
    m.I.update([6, 5])
    _verify(m.I, [5, 6])
    m.I = [0, -1, 1]
    _verify(m.I, [-1, 0, 1])