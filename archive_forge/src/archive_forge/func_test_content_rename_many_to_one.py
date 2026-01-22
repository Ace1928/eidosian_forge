from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_content_rename_many_to_one(self):
    blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
    blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
    blob3 = make_object(Blob, data=b'a\nb\nc\nf\n')
    tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
    tree2 = self.commit_tree([(b'c', blob3)])
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'c', F, blob3.id)), TreeChange.delete((b'b', F, blob2.id))], self.detect_renames(tree1, tree2))