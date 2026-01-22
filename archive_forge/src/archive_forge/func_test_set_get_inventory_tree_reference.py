from breezy.bzr import inventory, inventorytree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_set_get_inventory_tree_reference(self):
    """This tests that setting a tree reference is persistent."""
    tree = self.make_branch_and_tree('.')
    if not isinstance(tree, inventorytree.InventoryTree):
        raise TestNotApplicable('not an inventory tree')
    transform = tree.transform()
    trans_id = transform.new_directory('reference', transform.root, b'subtree-id')
    transform.set_tree_reference(b'subtree-revision', trans_id)
    transform.apply()
    tree = tree.controldir.open_workingtree()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(b'subtree-revision', tree.root_inventory.get_entry(b'subtree-id').reference_revision)