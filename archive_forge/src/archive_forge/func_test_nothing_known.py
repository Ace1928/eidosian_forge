from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_nothing_known(self):
    graph = self.make_graph(ancestry_1)
    self.assertFindDistance(0, graph, NULL_REVISION, [])
    self.assertFindDistance(1, graph, b'rev1', [])
    self.assertFindDistance(2, graph, b'rev2a', [])
    self.assertFindDistance(2, graph, b'rev2b', [])
    self.assertFindDistance(3, graph, b'rev3', [])
    self.assertFindDistance(4, graph, b'rev4', [])