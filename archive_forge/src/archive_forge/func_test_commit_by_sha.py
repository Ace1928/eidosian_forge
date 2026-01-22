from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_commit_by_sha(self):
    r = MemoryRepo()
    [c1] = build_commit_graph(r.object_store, [[1]])
    self.assertEqual(c1, parse_commit(r, c1.id))