from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference_complex_shortcut(self):
    graph = self.make_graph(complex_shortcut)
    self.assertEqual(({b'm', b'i', b'e'}, {b'n', b'h'}), graph.find_difference(b'm', b'n'))