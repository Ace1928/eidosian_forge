from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_follow_rename_remove_path(self):
    blob = make_object(Blob, data=b'blob')
    _, _, _, c4, c5, c6 = self.make_linear_commits(6, trees={1: [(b'a', blob), (b'c', blob)], 2: [], 3: [], 4: [(b'b', blob)], 5: [(b'a', blob)], 6: [(b'c', blob)]})

    def e(n):
        return (n, F, blob.id)
    self.assertWalkYields([TestWalkEntry(c6, [TreeChange(CHANGE_RENAME, e(b'a'), e(b'c'))]), TestWalkEntry(c5, [TreeChange(CHANGE_RENAME, e(b'b'), e(b'a'))]), TestWalkEntry(c4, [TreeChange.add(e(b'b'))])], [c6.id], paths=[b'c'], follow=True)