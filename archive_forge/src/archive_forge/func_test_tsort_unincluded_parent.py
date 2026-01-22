import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_tsort_unincluded_parent(self):
    """Sort nodes, but don't include some parents in the output"""
    self.assertSortAndIterate([(0, [1]), (1, [2])], [1, 0])