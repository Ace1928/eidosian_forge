import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_topo_sort_cycle_with_tail(self):
    """TopoSort traps graph with longer cycle"""
    self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [2], 2: [3, 4], 3: [0], 4: []}.items())