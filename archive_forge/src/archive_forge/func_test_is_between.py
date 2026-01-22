from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_is_between(self):
    graph = self.make_graph(ancestry_1)
    self.assertEqual(True, graph.is_between(b'null:', b'null:', b'null:'))
    self.assertEqual(True, graph.is_between(b'rev1', b'null:', b'rev1'))
    self.assertEqual(True, graph.is_between(b'rev1', b'rev1', b'rev4'))
    self.assertEqual(True, graph.is_between(b'rev4', b'rev1', b'rev4'))
    self.assertEqual(True, graph.is_between(b'rev3', b'rev1', b'rev4'))
    self.assertEqual(False, graph.is_between(b'rev4', b'rev1', b'rev3'))
    self.assertEqual(False, graph.is_between(b'rev1', b'rev2a', b'rev4'))
    self.assertEqual(False, graph.is_between(b'null:', b'rev1', b'rev4'))