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
def test_flattener(self):
    tup = (1, 0, (0, 1), (2, 3))
    self.assertEqual((1, 0, 0, 1, 2, 3), flatten_tuple(tup))
    li = [0]
    self.assertEqual((0,), flatten_tuple(li))
    ex = [(1, 0), [2, 3]]
    self.assertEqual((1, 0, 2, 3), flatten_tuple(ex))