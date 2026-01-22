from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_rewrite_threshold(self):
    blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
    blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
    blob3 = make_object(Blob, data=b'a\nb\nf\ng\n')
    tree1 = self.commit_tree([(b'a', blob1)])
    tree2 = self.commit_tree([(b'a', blob3), (b'b', blob2)])
    no_renames = [TreeChange(CHANGE_MODIFY, (b'a', F, blob1.id), (b'a', F, blob3.id)), TreeChange(CHANGE_COPY, (b'a', F, blob1.id), (b'b', F, blob2.id))]
    self.assertEqual(no_renames, self.detect_renames(tree1, tree2))
    self.assertEqual(no_renames, self.detect_renames(tree1, tree2, rewrite_threshold=40))
    self.assertEqual([TreeChange.add((b'a', F, blob3.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob2.id))], self.detect_renames(tree1, tree2, rewrite_threshold=80))