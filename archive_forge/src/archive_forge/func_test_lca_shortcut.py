from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_lca_shortcut(self):
    """Test least-common ancestor on this history shortcut"""
    graph = self.make_graph(history_shortcut)
    self.assertEqual({b'rev2b'}, graph.find_lca(b'rev3a', b'rev3b'))