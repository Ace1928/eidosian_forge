from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_racing_shortcuts(self):
    graph = self.make_graph(racing_shortcuts)
    self.assertFindUniqueAncestors(graph, [b'p', b'q', b'z'], b'z', [b'y'])
    self.assertFindUniqueAncestors(graph, [b'h', b'i', b'j', b'y'], b'j', [b'z'])