import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_mainline_revs_partial(self):
    self.assertSortAndIterate({'A': ['E', 'B'], 'B': ['D', 'C'], 'C': ['D'], 'D': ['E'], 'E': []}, 'A', [(0, 'A', 0, False), (1, 'B', 0, False), (2, 'C', 1, True)], False, mainline_revisions=['D', 'B', 'A'])
    self.assertSortAndIterate({'A': ['E', 'B'], 'B': ['D', 'C'], 'C': ['D'], 'D': ['E'], 'E': []}, 'A', [(0, 'A', 0, (4,), False), (1, 'B', 0, (3,), False), (2, 'C', 1, (2, 1, 1), True)], True, mainline_revisions=['D', 'B', 'A'])