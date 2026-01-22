from breezy.bzr import inventory, inventorytree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_may_not_autodetect_subtree(self):
    tree = self.prepare_with_subtree()
    self.assertIn(tree.kind('subtree'), ('directory', 'tree-reference'))