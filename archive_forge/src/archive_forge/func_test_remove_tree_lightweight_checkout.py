import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_lightweight_checkout(self):
    self.tree.branch.create_checkout('branch2', lightweight=True)
    self.assertPathExists('branch2/foo')
    output = self.run_bzr_error(['You cannot remove the working tree from a lightweight checkout'], 'remove-tree', retcode=3, working_dir='branch2')
    self.assertPathExists('branch2/foo')
    self.assertPathExists('branch1/foo')