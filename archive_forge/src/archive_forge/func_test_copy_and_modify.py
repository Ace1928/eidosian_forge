import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_copy_and_modify(self):
    a = Blob.from_string(b'a')
    b = Blob.from_string(b'b')
    delta = self.transform([('copy', (b'a', stat.S_IFREG | 420, a), (b'b', stat.S_IFREG | 420, b))])
    expected_delta = TreeDelta()
    expected_delta.copied.append(TreeChange(b'git:b', ('a', 'b'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'b'), ('file', 'file'), (False, False), True))
    self.assertEqual(expected_delta, delta)