import collections.abc
import pickle
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.suffix import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.constraint import constraint, constraint_list
from pyomo.core.kernel.block import block, block_dict
def test_clear_value(self):
    x = variable()
    s = suffix()
    s[x] = 1.0
    self.assertEqual(len(s), 1)
    s.clear_value(x)
    self.assertEqual(len(s), 0)
    s.clear_value(x)