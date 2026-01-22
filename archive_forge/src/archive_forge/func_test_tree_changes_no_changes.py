from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_no_changes(self):
    blob = make_object(Blob, data=b'blob')
    tree = self.commit_tree([(b'a', blob), (b'b/c', blob)])
    self.assertChangesEqual([], self.empty_tree, self.empty_tree)
    self.assertChangesEqual([], tree, tree)
    self.assertChangesEqual([TreeChange(CHANGE_UNCHANGED, (b'a', F, blob.id), (b'a', F, blob.id)), TreeChange(CHANGE_UNCHANGED, (b'b/c', F, blob.id), (b'b/c', F, blob.id))], tree, tree, want_unchanged=True)