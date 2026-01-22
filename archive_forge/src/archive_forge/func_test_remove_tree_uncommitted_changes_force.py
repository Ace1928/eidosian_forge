import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_uncommitted_changes_force(self):
    self.build_tree(['branch1/bar'])
    self.tree.add('bar')
    self.run_bzr('remove-tree branch1 --force')
    self.assertPathDoesNotExist('branch1/foo')
    self.assertPathExists('branch1/bar')