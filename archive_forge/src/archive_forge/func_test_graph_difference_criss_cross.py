from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference_criss_cross(self):
    graph = self.make_graph(criss_cross)
    self.assertEqual(({b'rev3a'}, {b'rev3b'}), graph.find_difference(b'rev3a', b'rev3b'))
    self.assertEqual((set(), {b'rev3b', b'rev2b'}), graph.find_difference(b'rev2a', b'rev3b'))