from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_changes_with_renames(self):
    blob = make_object(Blob, data=b'blob')
    c1, c2 = self.make_linear_commits(2, trees={1: [(b'a', blob)], 2: [(b'b', blob)]})
    entry_a = (b'a', F, blob.id)
    entry_b = (b'b', F, blob.id)
    changes_without_renames = [TreeChange.delete(entry_a), TreeChange.add(entry_b)]
    changes_with_renames = [TreeChange(CHANGE_RENAME, entry_a, entry_b)]
    self.assertWalkYields([TestWalkEntry(c2, changes_without_renames)], [c2.id], max_entries=1)
    detector = RenameDetector(self.store)
    self.assertWalkYields([TestWalkEntry(c2, changes_with_renames)], [c2.id], max_entries=1, rename_detector=detector)