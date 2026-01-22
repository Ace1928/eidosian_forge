from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_tagged_tag(self):
    self.assertMissingMatch([self._normal_tag.id], [self._tag_of_tag.id], [self._tag_of_tag.id])
    self.assertMissingMatch([self.cmt(1).id], [self._tag_of_tag.id], [self._normal_tag.id, self._tag_of_tag.id])