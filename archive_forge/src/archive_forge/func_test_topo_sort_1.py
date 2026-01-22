import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_topo_sort_1(self):
    """TopoSort simple nontrivial graph"""
    self.assertTopoSortOrder({0: [3], 1: [4], 2: [1, 4], 3: [], 4: [0, 3]})