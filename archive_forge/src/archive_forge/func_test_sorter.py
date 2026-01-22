import pyomo.common.unittest as unittest
from pyomo.core.base.enums import SortComponents
def test_sorter(self):
    self.assertEqual(SortComponents.sorter(), SortComponents.UNSORTED)
    self.assertEqual(SortComponents.sorter(True, False), SortComponents.ALPHABETICAL)
    self.assertEqual(SortComponents.sorter(False, True), SortComponents.SORTED_INDICES)
    self.assertEqual(SortComponents.sorter(True, True), SortComponents.ALPHABETICAL | SortComponents.SORTED_INDICES)