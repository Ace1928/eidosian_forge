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
def test_getset_direction(self):
    s = suffix()
    s.set_direction(suffix.LOCAL)
    self.assertEqual(s.get_direction(), suffix.LOCAL)
    s.set_direction(suffix.IMPORT)
    self.assertEqual(s.get_direction(), suffix.IMPORT)
    s.set_direction(suffix.EXPORT)
    self.assertEqual(s.get_direction(), suffix.EXPORT)
    s.set_direction(suffix.IMPORT_EXPORT)
    self.assertEqual(s.get_direction(), suffix.IMPORT_EXPORT)
    with self.assertRaises(ValueError):
        s.set_direction('export')