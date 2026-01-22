from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_bogus_haves(self):
    """Ensure non-existent SHA in haves are tolerated."""
    bogus_sha = self.cmt(2).id[::-1]
    haves = [self.cmt(1).id, bogus_sha]
    wants = [self.cmt(3).id]
    self.assertMissingMatch(haves, wants, self.missing_1_3)