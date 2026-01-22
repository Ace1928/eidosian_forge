from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_known_in_ancestry(self):
    graph = self.make_graph(ancestry_1)
    self.assertFindDistance(2, graph, b'rev2a', [(b'rev1', 1)])
    self.assertFindDistance(3, graph, b'rev3', [(b'rev2a', 2)])