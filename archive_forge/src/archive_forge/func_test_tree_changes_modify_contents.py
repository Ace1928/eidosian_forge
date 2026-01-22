from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_modify_contents(self):
    blob_a1 = make_object(Blob, data=b'a1')
    blob_a2 = make_object(Blob, data=b'a2')
    tree1 = self.commit_tree([(b'a', blob_a1)])
    tree2 = self.commit_tree([(b'a', blob_a2)])
    self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob_a1.id), (b'a', F, blob_a2.id))], tree1, tree2)