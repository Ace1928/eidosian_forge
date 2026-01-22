from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_filter_with_merge(self):
    blob_a = make_object(Blob, data=b'a')
    blob_a2 = make_object(Blob, data=b'a2')
    blob_b = make_object(Blob, data=b'b')
    blob_b2 = make_object(Blob, data=b'b2')
    x1, y2, m3 = self.make_commits([[1], [2], [3, 1, 2]], trees={1: [(b'x/a', blob_a)], 2: [(b'y/b', blob_b)], 3: [(b'x/a', blob_a2), (b'y/b', blob_b2)]})
    walker = Walker(self.store, m3.id)
    entries = list(walker)
    walker_entry = entries[0]
    self.assertEqual(walker_entry.commit.id, m3.id)
    changes = walker_entry.changes(b'x')
    self.assertEqual(1, len(changes))
    entry_a = (b'a', F, blob_a.id)
    entry_a2 = (b'a', F, blob_a2.id)
    self.assertEqual([[TreeChange(CHANGE_MODIFY, entry_a, entry_a2)]], changes)