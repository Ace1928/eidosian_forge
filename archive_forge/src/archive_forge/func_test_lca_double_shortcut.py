from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_lca_double_shortcut(self):
    graph = self.make_graph(double_shortcut)
    self.assertEqual(b'c', graph.find_unique_lca(b'f', b'g'))