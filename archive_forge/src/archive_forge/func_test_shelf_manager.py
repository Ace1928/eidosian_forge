from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import ShelvingUnsupported
def test_shelf_manager(self):
    tree = self.make_branch_and_tree('.')
    if self.workingtree_format.supports_store_uncommitted:
        self.assertIsNot(None, tree.get_shelf_manager())
    else:
        self.assertRaises(ShelvingUnsupported, tree.get_shelf_manager)