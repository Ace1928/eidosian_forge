import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_heads_linear(self):
    graph = self.make_known_graph(test_graph.racing_shortcuts)
    self.assertEqual({b'w'}, graph.heads([b'w', b's']))
    self.assertEqual({b'z'}, graph.heads([b'w', b's', b'z']))
    self.assertEqual({b'w', b'q'}, graph.heads([b'w', b's', b'q']))
    self.assertEqual({b'z'}, graph.heads([b's', b'z']))