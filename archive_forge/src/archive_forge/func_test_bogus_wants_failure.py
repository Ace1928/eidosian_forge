from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_bogus_wants_failure(self):
    """Ensure non-existent SHA in wants are not tolerated."""
    bogus_sha = self.cmt(2).id[::-1]
    haves = [self.cmt(1).id]
    wants = [self.cmt(3).id, bogus_sha]
    self.assertRaises(KeyError, MissingObjectFinder, self.store, haves, wants, shallow=set())