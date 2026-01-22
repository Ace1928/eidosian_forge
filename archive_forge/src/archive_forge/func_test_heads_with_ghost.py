import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_heads_with_ghost(self):
    graph = self.make_known_graph(test_graph.with_ghost)
    self.assertEqual({b'e', b'g'}, graph.heads([b'e', b'g']))
    self.assertEqual({b'a', b'c'}, graph.heads([b'a', b'c']))
    self.assertEqual({b'a', b'g'}, graph.heads([b'a', b'g']))
    self.assertEqual({b'f', b'g'}, graph.heads([b'f', b'g']))
    self.assertEqual({b'c'}, graph.heads([b'c', b'g']))
    self.assertEqual({b'c'}, graph.heads([b'c', b'b', b'd', b'g']))
    self.assertEqual({b'a', b'c'}, graph.heads([b'a', b'c', b'e', b'g']))
    self.assertEqual({b'a', b'c'}, graph.heads([b'a', b'c', b'f']))