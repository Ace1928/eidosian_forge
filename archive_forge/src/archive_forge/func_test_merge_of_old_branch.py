from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
@expectedFailure
def test_merge_of_old_branch(self):
    self.maxDiff = None
    c1, c2, c3, c4, c5 = self.make_commits([[1], [2, 1], [3, 2], [4, 1], [5, 3, 4]], times=[1, 3, 4, 2, 5])
    self.assertWalkYields([c5, c4, c3, c2, c1], [c5.id])
    self.assertWalkYields([c3, c2, c1], [c3.id])
    self.assertWalkYields([c2, c1], [c2.id])