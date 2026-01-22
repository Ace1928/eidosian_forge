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
def test_set_all_values(self):
    x = variable()
    y = variable()
    s = suffix()
    s[x] = 1.0
    s[y] = None
    self.assertEqual(s[x], 1.0)
    self.assertEqual(s[y], None)
    s.set_all_values(0)
    self.assertEqual(s[x], 0)
    self.assertEqual(s[y], 0)