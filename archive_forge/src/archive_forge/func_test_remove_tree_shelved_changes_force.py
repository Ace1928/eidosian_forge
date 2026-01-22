import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_shelved_changes_force(self):
    tree = self.make_branch_and_tree('.')
    creator = shelf.ShelfCreator(tree, tree.basis_tree(), [])
    self.addCleanup(creator.finalize)
    shelf_id = tree.get_shelf_manager().shelve_changes(creator, 'Foo')
    self.run_bzr('remove-tree --force')
    self.run_bzr('checkout')
    self.assertIs(None, tree.get_shelf_manager().last_shelf())