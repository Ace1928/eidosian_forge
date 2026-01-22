from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_is_ancestor(self):
    graph = self.make_graph(ancestry_1)
    self.assertEqual(True, graph.is_ancestor(b'null:', b'null:'))
    self.assertEqual(True, graph.is_ancestor(b'null:', b'rev1'))
    self.assertEqual(False, graph.is_ancestor(b'rev1', b'null:'))
    self.assertEqual(True, graph.is_ancestor(b'null:', b'rev4'))
    self.assertEqual(False, graph.is_ancestor(b'rev4', b'null:'))
    self.assertEqual(False, graph.is_ancestor(b'rev4', b'rev2b'))
    self.assertEqual(True, graph.is_ancestor(b'rev2b', b'rev4'))
    self.assertEqual(False, graph.is_ancestor(b'rev2b', b'rev3'))
    self.assertEqual(False, graph.is_ancestor(b'rev3', b'rev2b'))
    instrumented_provider = InstrumentedParentsProvider(graph)
    instrumented_graph = _mod_graph.Graph(instrumented_provider)
    instrumented_graph.is_ancestor(b'rev2a', b'rev2b')
    self.assertTrue(b'null:' not in instrumented_provider.calls)