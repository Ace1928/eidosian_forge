from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_multiple_revisions(self):
    graph = self.make_graph(ancestry_1)
    self.assertFindUniqueAncestors(graph, [b'rev4'], b'rev4', [b'rev3', b'rev2b'])
    self.assertFindUniqueAncestors(graph, [b'rev2a', b'rev3', b'rev4'], b'rev4', [b'rev2b'])