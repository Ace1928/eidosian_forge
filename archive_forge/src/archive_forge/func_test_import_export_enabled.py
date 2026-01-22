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
def test_import_export_enabled(self):
    s = suffix()
    s.direction = suffix.LOCAL
    self.assertEqual(s.direction, suffix.LOCAL)
    self.assertEqual(s.export_enabled(), False)
    self.assertEqual(s.import_enabled(), False)
    s.direction = suffix.IMPORT
    self.assertEqual(s.direction, suffix.IMPORT)
    self.assertEqual(s.export_enabled(), False)
    self.assertEqual(s.import_enabled(), True)
    s.direction = suffix.EXPORT
    self.assertEqual(s.direction, suffix.EXPORT)
    self.assertEqual(s.export_enabled(), True)
    self.assertEqual(s.import_enabled(), False)
    s.direction = suffix.IMPORT_EXPORT
    self.assertEqual(s.direction, suffix.IMPORT_EXPORT)
    self.assertEqual(s.export_enabled(), True)
    self.assertEqual(s.import_enabled(), True)
    with self.assertRaises(ValueError):
        s.direction = 'export'