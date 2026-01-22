from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_add_delete(self):
    blob_a = make_object(Blob, data=b'a')
    blob_b = make_object(Blob, data=b'b')
    tree = self.commit_tree([(b'a', blob_a, 33188), (b'x/b', blob_b, 33261)])
    self.assertChangesEqual([TreeChange.add((b'a', 33188, blob_a.id)), TreeChange.add((b'x/b', 33261, blob_b.id))], self.empty_tree, tree)
    self.assertChangesEqual([TreeChange.delete((b'a', 33188, blob_a.id)), TreeChange.delete((b'x/b', 33261, blob_b.id))], tree, self.empty_tree)