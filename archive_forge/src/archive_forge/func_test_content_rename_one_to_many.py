from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_content_rename_one_to_many(self):
    blob1 = make_object(Blob, data=b'aa\nb\nc\nd\ne\n')
    blob2 = make_object(Blob, data=b'ab\nb\nc\nd\ne\n')
    blob3 = make_object(Blob, data=b'aa\nb\nc\nd\nf\n')
    tree1 = self.commit_tree([(b'a', blob1)])
    tree2 = self.commit_tree([(b'b', blob2), (b'c', blob3)])
    self.assertEqual([TreeChange(CHANGE_COPY, (b'a', F, blob1.id), (b'b', F, blob2.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'c', F, blob3.id))], self.detect_renames(tree1, tree2))