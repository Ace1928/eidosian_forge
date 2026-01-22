from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from .utils import build_commit_graph, make_object
def test_attrs(self):
    c1, c2 = build_commit_graph(self.store, [[1], [2, 1]], attrs={1: {'message': b'Hooray!'}})
    self.assertEqual(b'Hooray!', c1.message)
    self.assertEqual(b'Commit 2', c2.message)