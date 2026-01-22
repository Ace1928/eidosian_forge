import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_unversioned(self):
    b = Blob.from_string(b'b')
    delta = self.transform([('add', (None, None, None), (b'a', stat.S_IFREG | 420, b))], target_extras={b'a'})
    expected_delta = TreeDelta()
    expected_delta.unversioned.append(TreeChange(None, (None, 'a'), True, (False, False), (None, b'TREE_ROOT'), (None, 'a'), (None, 'file'), (None, False), False))
    self.assertEqual(delta, expected_delta)
    delta = self.transform([('add', (b'a', stat.S_IFREG | 420, b), (b'a', stat.S_IFREG | 420, b))], source_extras={b'a'}, target_extras={b'a'})
    expected_delta = TreeDelta()
    expected_delta.unversioned.append(TreeChange(None, ('a', 'a'), False, (False, False), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', 'file'), (False, False), False))
    self.assertEqual(delta, expected_delta)