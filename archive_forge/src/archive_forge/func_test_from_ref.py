from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_from_ref(self):
    r = MemoryRepo()
    c1, c2, c3 = build_commit_graph(r.object_store, [[1], [2, 1], [3, 1, 2]])
    r.refs[b'refs/heads/foo'] = c1.id
    self.assertEqual(r[c1.tree], parse_tree(r, b'foo'))