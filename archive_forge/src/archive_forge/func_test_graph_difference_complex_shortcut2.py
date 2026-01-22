from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_graph_difference_complex_shortcut2(self):
    graph = self.make_graph(complex_shortcut2)
    self.assertEqual(({b't'}, {b'j', b'u'}), graph.find_difference(b't', b'u'))