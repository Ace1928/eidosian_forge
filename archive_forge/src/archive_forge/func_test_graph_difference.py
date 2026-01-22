from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference(self):
    graph = self.make_graph(ancestry_1)
    self.assertEqual((set(), set()), graph.find_difference(b'rev1', b'rev1'))
    self.assertEqual((set(), {b'rev1'}), graph.find_difference(NULL_REVISION, b'rev1'))
    self.assertEqual(({b'rev1'}, set()), graph.find_difference(b'rev1', NULL_REVISION))
    self.assertEqual(({b'rev2a', b'rev3'}, {b'rev2b'}), graph.find_difference(b'rev3', b'rev2b'))
    self.assertEqual(({b'rev4', b'rev3', b'rev2a'}, set()), graph.find_difference(b'rev4', b'rev2b'))