from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_iter_ancestry_with_ghost(self):
    graph = self.make_graph(with_ghost)
    expected = with_ghost.copy()
    expected[b'g'] = None
    self.assertEqual(expected, dict(graph.iter_ancestry([b'a', b'c'])))
    expected.pop(b'a')
    self.assertEqual(expected, dict(graph.iter_ancestry([b'c'])))