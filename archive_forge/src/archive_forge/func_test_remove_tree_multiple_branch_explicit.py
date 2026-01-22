import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_multiple_branch_explicit(self):
    self.tree.controldir.sprout('branch2')
    self.run_bzr('remove-tree branch1 branch2')
    self.assertPathDoesNotExist('branch1/foo')
    self.assertPathDoesNotExist('branch2/foo')