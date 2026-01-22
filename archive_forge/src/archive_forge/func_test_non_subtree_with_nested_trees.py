import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_non_subtree_with_nested_trees(self):
    tree = self.make_branch_and_tree('.', format='dirstate')
    self.assertFalse(tree.supports_tree_reference())
    self.build_tree(['dir/'])
    tree.set_root_id(b'root')
    tree.add(['dir'], ids=[b'dir-id'])
    self.make_branch_and_tree('dir')
    self.assertEqual('directory', tree.kind('dir'))
    tree.lock_read()
    expected = [(b'dir-id', (None, 'dir'), True, (False, True), (None, b'root'), (None, 'dir'), (None, 'directory'), (None, False), False), (b'root', (None, ''), True, (False, True), (None, None), (None, ''), (None, 'directory'), (None, False), False)]
    self.assertEqual(expected, list(tree.iter_changes(tree.basis_tree(), specific_files=['dir'])))
    tree.unlock()
    tree.commit('first post')
    os.rename('dir', 'also-dir')
    tree.lock_read()
    expected = [(b'dir-id', ('dir', 'dir'), True, (True, True), (b'root', b'root'), ('dir', 'dir'), ('directory', None), (False, False), False)]
    self.assertEqual(expected, list(tree.iter_changes(tree.basis_tree())))
    tree.unlock()