from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_since_over_scan(self):
    commits = self.make_linear_commits(11, times=[9, 0, 1, 2, 3, 4, 5, 8, 6, 7, 9])
    c8, _, c10, c11 = commits[-4:]
    del self.store[commits[0].id]
    self.assertWalkYields([c11, c10, c8], [c11.id], since=7)