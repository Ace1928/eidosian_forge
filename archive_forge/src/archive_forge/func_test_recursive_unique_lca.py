from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_recursive_unique_lca(self):
    """Test finding a unique least common ancestor.

        ancestry_1 should always have a single common ancestor
        """
    graph = self.make_graph(ancestry_1)
    self.assertEqual(NULL_REVISION, graph.find_unique_lca(NULL_REVISION, NULL_REVISION))
    self.assertEqual(NULL_REVISION, graph.find_unique_lca(NULL_REVISION, b'rev1'))
    self.assertEqual(b'rev1', graph.find_unique_lca(b'rev1', b'rev1'))
    self.assertEqual(b'rev1', graph.find_unique_lca(b'rev2a', b'rev2b'))
    self.assertEqual((b'rev1', 1), graph.find_unique_lca(b'rev2a', b'rev2b', count_steps=True))