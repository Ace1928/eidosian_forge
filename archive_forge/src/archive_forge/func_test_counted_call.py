import functools
import pickle
import platform
import sys
import types
import pyomo.common.unittest as unittest
from pyomo.common.config import ConfigValue, ConfigList, ConfigDict
from pyomo.common.dependencies import (
from pyomo.core.base.util import flatten_tuple
from pyomo.core.base.initializer import (
from pyomo.environ import ConcreteModel, Var
def test_counted_call(self):

    def x_init(m, i):
        return i + 1

    def y_init(m, i, j):
        return j * (i + 1)

    def z_init(m, i, j, k):
        return i * 100 + j * 10 + k

    def bogus(m, i, j):
        return None
    m = ConcreteModel()
    m.x = Var([1, 2, 3])
    a = Initializer(x_init)
    b = CountedCallInitializer(m.x, a)
    self.assertIs(type(b), CountedCallInitializer)
    self.assertFalse(b.constant())
    self.assertFalse(b.verified)
    self.assertFalse(b.contains_indices())
    self.assertFalse(b._scalar)
    self.assertIs(a._fcn, b._fcn)
    c = b(None, 1)
    self.assertIs(type(c), int)
    self.assertEqual(c, 2)
    a = Initializer(bogus)
    b = CountedCallInitializer(m.x, a)
    self.assertIs(type(b), CountedCallInitializer)
    self.assertFalse(b.constant())
    self.assertFalse(b.verified)
    self.assertFalse(b.contains_indices())
    self.assertFalse(b._scalar)
    self.assertIs(a._fcn, b._fcn)
    c = b(None, 1)
    self.assertIs(type(c), CountedCallGenerator)
    with self.assertRaisesRegex(ValueError, 'Counted Var rule returned None'):
        next(c)
    a = Initializer(y_init)
    b = CountedCallInitializer(m.x, a)
    self.assertIs(type(b), CountedCallInitializer)
    self.assertFalse(b.constant())
    self.assertFalse(b.verified)
    self.assertFalse(b.contains_indices())
    self.assertFalse(b._scalar)
    self.assertIs(a._fcn, b._fcn)
    c = b(None, 1)
    self.assertIs(type(c), CountedCallGenerator)
    self.assertEqual(next(c), 2)
    self.assertEqual(next(c), 3)
    self.assertEqual(next(c), 4)
    m.y = Var([(1, 2), (3, 5)])
    a = Initializer(y_init)
    b = CountedCallInitializer(m.y, a)
    self.assertIs(type(b), CountedCallInitializer)
    self.assertFalse(b.constant())
    self.assertFalse(b.verified)
    self.assertFalse(b.contains_indices())
    self.assertFalse(b._scalar)
    self.assertIs(a._fcn, b._fcn)
    c = b(None, (3, 5))
    self.assertIs(type(c), int)
    self.assertEqual(c, 20)
    a = Initializer(z_init)
    b = CountedCallInitializer(m.y, a)
    self.assertIs(type(b), CountedCallInitializer)
    self.assertFalse(b.constant())
    self.assertFalse(b.verified)
    self.assertFalse(b.contains_indices())
    self.assertFalse(b._scalar)
    self.assertIs(a._fcn, b._fcn)
    c = b(None, (3, 5))
    self.assertIs(type(c), CountedCallGenerator)
    self.assertEqual(next(c), 135)
    self.assertEqual(next(c), 235)
    self.assertEqual(next(c), 335)