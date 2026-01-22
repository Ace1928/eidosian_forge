import pyomo.common.unittest as unittest
from pyomo.core.base.enums import SortComponents
def test_mappings(self):
    self.assertEqual(SortComponents(True), SortComponents.SORTED_INDICES | SortComponents.ALPHABETICAL)
    self.assertEqual(SortComponents(False), SortComponents.UNSORTED)
    self.assertEqual(SortComponents(None), SortComponents.UNSORTED)
    with self.assertRaisesRegex(ValueError, '(999 is not a valid SortComponents)|(invalid value 999)'):
        SortComponents(999)
    self.assertEqual(SortComponents._missing_(False), SortComponents.UNSORTED)