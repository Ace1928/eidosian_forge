import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_heads_single(self):
    graph = self.make_known_graph(test_graph.ancestry_1)
    self.assertEqual({b'rev4'}, graph.heads([b'null:', b'rev4']))
    self.assertEqual({b'rev2a'}, graph.heads([b'rev1', b'rev2a']))
    self.assertEqual({b'rev2b'}, graph.heads([b'rev1', b'rev2b']))
    self.assertEqual({b'rev3'}, graph.heads([b'rev1', b'rev3']))
    self.assertEqual({b'rev3'}, graph.heads([b'rev3', b'rev2a']))
    self.assertEqual({b'rev4'}, graph.heads([b'rev1', b'rev4']))
    self.assertEqual({b'rev4'}, graph.heads([b'rev2a', b'rev4']))
    self.assertEqual({b'rev4'}, graph.heads([b'rev2b', b'rev4']))
    self.assertEqual({b'rev4'}, graph.heads([b'rev3', b'rev4']))