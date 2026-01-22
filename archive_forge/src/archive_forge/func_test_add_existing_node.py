import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_add_existing_node(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    self.assertGDFO(graph, b'rev4', 5)
    graph.add_node(b'rev4', [b'rev3', b'rev2b'])
    self.assertGDFO(graph, b'rev4', 5)
    graph.add_node(b'rev4', (b'rev3', b'rev2b'))