from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_lefthand_distance_ghosts(self):
    """A simple does it work test for graph.lefthand_distance(keys)."""
    nodes = {b'nonghost': [NULL_REVISION], b'toghost': [b'ghost']}
    graph = self.make_graph(nodes)
    distance_graph = graph.find_lefthand_distances([b'nonghost', b'toghost'])
    self.assertEqual({b'nonghost': 1, b'toghost': -1}, distance_graph)