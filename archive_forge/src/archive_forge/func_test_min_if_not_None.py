import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
def test_min_if_not_None(self):
    self.assertEqual(min_if_not_None([1, 2, None, 3, None]), 1)
    self.assertEqual(min_if_not_None([None, None, None]), None)
    self.assertEqual(min_if_not_None([]), None)
    self.assertEqual(min_if_not_None([None, 3, -1, 2]), -1)
    self.assertEqual(min_if_not_None([0]), 0)
    self.assertEqual(min_if_not_None([0, None]), 0)