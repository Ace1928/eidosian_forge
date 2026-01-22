from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_is_ancestor_boundary(self):
    """Ensure that we avoid searching the whole graph.

        This requires searching through b as a common ancestor, so we
        can identify that e is common.
        """
    graph = self.make_graph(boundary)
    instrumented_provider = InstrumentedParentsProvider(graph)
    graph = _mod_graph.Graph(instrumented_provider)
    self.assertFalse(graph.is_ancestor(b'a', b'c'))
    self.assertTrue(b'null:' not in instrumented_provider.calls)