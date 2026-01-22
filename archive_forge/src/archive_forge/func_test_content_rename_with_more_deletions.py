from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_content_rename_with_more_deletions(self):
    blob1 = make_object(Blob, data=b'')
    tree1 = self.commit_tree([(b'a', blob1), (b'b', blob1), (b'c', blob1), (b'd', blob1)])
    tree2 = self.commit_tree([(b'e', blob1), (b'f', blob1), (b'g', blob1)])
    self.maxDiff = None
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'e', F, blob1.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob1.id), (b'f', F, blob1.id)), TreeChange(CHANGE_RENAME, (b'c', F, blob1.id), (b'g', F, blob1.id)), TreeChange.delete((b'd', F, blob1.id))], self.detect_renames(tree1, tree2))