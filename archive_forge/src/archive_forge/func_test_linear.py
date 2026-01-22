from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_linear(self):
    c1, c2, c3 = self.make_linear_commits(3)
    self.assertWalkYields([c1], [c1.id])
    self.assertWalkYields([c2, c1], [c2.id])
    self.assertWalkYields([c3, c2, c1], [c3.id])
    self.assertWalkYields([c3, c2, c1], [c3.id, c1.id])
    self.assertWalkYields([c3, c2], [c3.id], exclude=[c1.id])
    self.assertWalkYields([c3, c2], [c3.id, c1.id], exclude=[c1.id])
    self.assertWalkYields([c3], [c3.id, c1.id], exclude=[c2.id])