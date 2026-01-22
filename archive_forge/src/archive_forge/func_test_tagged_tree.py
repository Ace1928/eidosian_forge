from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_tagged_tree(self):
    self.assertMissingMatch([], [self._tag_of_tree.id], [self._tag_of_tree.id, self.cmt(1).tree, self.f1_1_id])