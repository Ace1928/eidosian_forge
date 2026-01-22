import collections.abc
import pickle
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.variable import variable, variable_dict, variable_list
from pyomo.core.kernel.constraint import constraint, constraint_dict, constraint_list
from pyomo.core.kernel.objective import objective, objective_dict, objective_list
from pyomo.core.kernel.expression import expression, expression_dict, expression_list
from pyomo.core.kernel.block import block, block_dict, block_list
from pyomo.core.kernel.suffix import suffix
def test_getsetdelitem(self):
    cmap = ComponentMap()
    for c, val in self._components:
        self.assertTrue(c not in cmap)
    for c, val in self._components:
        cmap[c] = val
        self.assertEqual(cmap[c], val)
        self.assertEqual(cmap.get(c), val)
        del cmap[c]
        with self.assertRaises(KeyError):
            cmap[c]
        with self.assertRaises(KeyError):
            del cmap[c]
        self.assertEqual(cmap.get(c), None)