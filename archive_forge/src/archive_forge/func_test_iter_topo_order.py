from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_iter_topo_order(self):
    graph = self.make_graph(ancestry_1)
    args = [b'rev2a', b'rev3', b'rev1']
    topo_args = list(graph.iter_topo_order(args))
    self.assertEqual(set(args), set(topo_args))
    self.assertTrue(topo_args.index(b'rev2a') > topo_args.index(b'rev1'))
    self.assertTrue(topo_args.index(b'rev2a') < topo_args.index(b'rev3'))