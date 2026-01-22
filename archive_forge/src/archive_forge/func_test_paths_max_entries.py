from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_paths_max_entries(self):
    blob_a = make_object(Blob, data=b'a')
    blob_b = make_object(Blob, data=b'b')
    c1, c2 = self.make_linear_commits(2, trees={1: [(b'a', blob_a)], 2: [(b'a', blob_a), (b'b', blob_b)]})
    self.assertWalkYields([c2], [c2.id], paths=[b'b'], max_entries=1)
    self.assertWalkYields([c1], [c1.id], paths=[b'a'], max_entries=1)