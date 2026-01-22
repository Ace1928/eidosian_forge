from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test__remove_simple_descendants_disjoint(self):
    graph = self.make_graph(ancestry_1)
    self.assertRemoveDescendants({b'rev1', b'rev3'}, graph, {b'rev1', b'rev3'})