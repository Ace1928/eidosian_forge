from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_octopus_modify_conflict(self):
    r = list(range(5))
    parent_blobs = [make_object(Blob, data=bytes(i)) for i in r]
    merge_blob = make_object(Blob, data=b'merge')
    parents = [self.commit_tree([(b'a', parent_blobs[i])]) for i in r]
    merge = self.commit_tree([(b'a', merge_blob)])
    expected = [[TreeChange(CHANGE_MODIFY, (b'a', F, parent_blobs[i].id), (b'a', F, merge_blob.id)) for i in r]]
    self.assertChangesForMergeEqual(expected, parents, merge)