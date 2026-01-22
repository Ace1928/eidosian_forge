import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_revnos_are_globally_assigned(self):
    """revnos are assigned according to the revision they derive from."""
    self.assertSortAndIterate({'J': ['G', 'I'], 'I': ['H'], 'H': ['A'], 'G': ['D', 'F'], 'F': ['E'], 'E': ['A'], 'D': ['A', 'C'], 'C': ['B'], 'B': ['A'], 'A': []}.items(), 'J', [(0, 'J', 0, (4,), False), (1, 'I', 1, (1, 3, 2), False), (2, 'H', 1, (1, 3, 1), True), (3, 'G', 0, (3,), False), (4, 'F', 1, (1, 2, 2), False), (5, 'E', 1, (1, 2, 1), True), (6, 'D', 0, (2,), False), (7, 'C', 1, (1, 1, 2), False), (8, 'B', 1, (1, 1, 1), True), (9, 'A', 0, (1,), True)], True)