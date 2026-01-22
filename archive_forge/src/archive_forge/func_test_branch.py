from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_branch(self):
    c1, x2, x3, y4 = self.make_commits([[1], [2, 1], [3, 2], [4, 1]])
    self.assertWalkYields([x3, x2, c1], [x3.id])
    self.assertWalkYields([y4, c1], [y4.id])
    self.assertWalkYields([y4, x2, c1], [y4.id, x2.id])
    self.assertWalkYields([y4, x2], [y4.id, x2.id], exclude=[c1.id])
    self.assertWalkYields([y4, x3], [y4.id, x3.id], exclude=[x2.id])
    self.assertWalkYields([y4], [y4.id], exclude=[x3.id])
    self.assertWalkYields([x3, x2], [x3.id], exclude=[y4.id])