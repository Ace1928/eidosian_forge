from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_root(self):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        tree.add('')
        self.assertEqual([''], list(tree.all_versioned_paths()))
        if tree._format.supports_setting_file_ids:
            self.assertNotEqual(inventory.ROOT_ID, tree.path2id(''))