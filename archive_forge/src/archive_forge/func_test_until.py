from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_until(self):
    c1, c2, c3 = self.make_linear_commits(3)
    self.assertWalkYields([], [c3.id], until=-1)
    self.assertWalkYields([c1], [c3.id], until=0)
    self.assertWalkYields([c1], [c3.id], until=1)
    self.assertWalkYields([c1], [c3.id], until=99)
    self.assertWalkYields([c2, c1], [c3.id], until=100)
    self.assertWalkYields([c2, c1], [c3.id], until=101)
    self.assertWalkYields([c2, c1], [c3.id], until=199)
    self.assertWalkYields([c3, c2, c1], [c3.id], until=200)
    self.assertWalkYields([c3, c2, c1], [c3.id], until=201)
    self.assertWalkYields([c3, c2, c1], [c3.id], until=300)