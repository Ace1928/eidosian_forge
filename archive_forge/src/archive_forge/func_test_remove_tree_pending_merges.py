import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_pending_merges(self):
    self.run_bzr(['branch', 'branch1', 'branch2'])
    self.build_tree(['branch1/bar'])
    self.tree.add('bar')
    self.tree.commit('2')
    self.assertPathExists('branch1/bar')
    self.run_bzr(['merge', '../branch1'], working_dir='branch2')
    self.assertPathExists('branch2/bar')
    self.run_bzr(['revert', '.'], working_dir='branch2')
    self.assertPathDoesNotExist('branch2/bar')
    output = self.run_bzr_error(['Working tree .* has uncommitted changes'], 'remove-tree branch2', retcode=3)