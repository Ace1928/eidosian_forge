import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_sequence_numbers_increase_with_merges(self):
    self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A', 'B']}.items(), 'C', [(0, 'C', 0, False), (1, 'B', 1, True), (2, 'A', 0, True)], False)
    self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A', 'B']}.items(), 'C', [(0, 'C', 0, (2,), False), (1, 'B', 1, (1, 1, 1), True), (2, 'A', 0, (1,), True)], True)