from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_modify_no_conflict(self):
    blob1 = make_object(Blob, data=b'1')
    blob2 = make_object(Blob, data=b'2')
    parent1 = self.commit_tree([(b'a', blob1)])
    parent2 = merge = self.commit_tree([(b'a', blob2)])
    self.assertChangesForMergeEqual([], [parent1, parent2], merge)