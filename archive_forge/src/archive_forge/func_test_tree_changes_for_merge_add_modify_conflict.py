from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_add_modify_conflict(self):
    blob1 = make_object(Blob, data=b'1')
    blob2 = make_object(Blob, data=b'2')
    parent1 = self.commit_tree([])
    parent2 = self.commit_tree([(b'a', blob1)])
    merge = self.commit_tree([(b'a', blob2)])
    self.assertChangesForMergeEqual([[TreeChange.add((b'a', F, blob2.id)), TreeChange(CHANGE_MODIFY, (b'a', F, blob1.id), (b'a', F, blob2.id))]], [parent1, parent2], merge)