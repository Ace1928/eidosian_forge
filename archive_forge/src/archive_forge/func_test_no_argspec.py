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
def test_no_argspec(self):
    a = Initializer(getattr)
    self.assertIs(type(a), IndexedCallInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertFalse(a.contains_indices())
    self.assertEqual(a(None, '__class__'), type(None))
    basetwo = functools.partial(int)
    a = Initializer(basetwo)
    self.assertIs(type(a), IndexedCallInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertFalse(a.contains_indices())
    self.assertEqual(a('111', 2), 7)
    basetwo = functools.partial(int, '101', base=2)
    a = Initializer(basetwo)
    if is_pypy and sys.pypy_version_info[:3] >= (7, 3, 14):
        self.assertIs(type(a), ScalarCallInitializer)
        self.assertTrue(a.constant())
    else:
        self.assertIs(type(a), IndexedCallInitializer)
        self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertFalse(a.contains_indices())