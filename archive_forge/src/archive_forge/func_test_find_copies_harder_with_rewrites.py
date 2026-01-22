from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_find_copies_harder_with_rewrites(self):
    blob_a1 = make_object(Blob, data=b'a\nb\nc\nd\n')
    blob_a2 = make_object(Blob, data=b'f\ng\nh\ni\n')
    blob_b2 = make_object(Blob, data=b'a\nb\nc\ne\n')
    tree1 = self.commit_tree([(b'a', blob_a1)])
    tree2 = self.commit_tree([(b'a', blob_a2), (b'b', blob_b2)])
    self.assertEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob_a1.id), (b'a', F, blob_a2.id)), TreeChange(CHANGE_COPY, (b'a', F, blob_a1.id), (b'b', F, blob_b2.id))], self.detect_renames(tree1, tree2, find_copies_harder=True))
    self.assertEqual([TreeChange.add((b'a', F, blob_a2.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob_a1.id), (b'b', F, blob_b2.id))], self.detect_renames(tree1, tree2, rewrite_threshold=50, find_copies_harder=True))