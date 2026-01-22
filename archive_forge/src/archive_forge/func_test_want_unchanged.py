from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_want_unchanged(self):
    blob_a1 = make_object(Blob, data=b'a\nb\nc\nd\n')
    blob_b = make_object(Blob, data=b'b')
    blob_c2 = make_object(Blob, data=b'a\nb\nc\ne\n')
    tree1 = self.commit_tree([(b'a', blob_a1), (b'b', blob_b)])
    tree2 = self.commit_tree([(b'c', blob_c2), (b'b', blob_b)])
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'c', F, blob_c2.id))], self.detect_renames(tree1, tree2))
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'c', F, blob_c2.id)), TreeChange(CHANGE_UNCHANGED, (b'b', F, blob_b.id), (b'b', F, blob_b.id))], self.detect_renames(tree1, tree2, want_unchanged=True))