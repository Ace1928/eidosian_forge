from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_topo_reorder_multiple_parents(self):
    c1, c2, c3 = self.make_commits([[1], [2], [3, 1, 2]])
    self.assertTopoOrderEqual([c3, c2, c1], [c3, c2, c1])
    self.assertTopoOrderEqual([c3, c1, c2], [c3, c1, c2])
    self.assertTopoOrderEqual([c3, c2, c1], [c2, c3, c1])
    self.assertTopoOrderEqual([c3, c1, c2], [c1, c3, c2])
    self.assertTopoOrderEqual([c3, c2, c1], [c1, c2, c3])
    self.assertTopoOrderEqual([c3, c2, c1], [c2, c1, c3])