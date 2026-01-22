from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_paths_subtree(self):
    blob_a = make_object(Blob, data=b'a')
    blob_b = make_object(Blob, data=b'b')
    c1, c2, c3 = self.make_linear_commits(3, trees={1: [(b'x/a', blob_a)], 2: [(b'b', blob_b), (b'x/a', blob_a)], 3: [(b'b', blob_b), (b'x/a', blob_a), (b'x/b', blob_b)]})
    self.assertWalkYields([c2], [c3.id], paths=[b'b'])
    self.assertWalkYields([c3, c1], [c3.id], paths=[b'x'])