from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_1_to_2(self):
    self.assertMissingMatch([self.cmt(1).id], [self.cmt(2).id], self.missing_1_2)