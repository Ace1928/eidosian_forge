import os
from breezy.tests import TestCaseWithTransport
def test_find_null_merge_base(self):
    tree = self.make_branch_and_tree('foo')
    tree.commit('message')
    tree2 = self.make_branch_and_tree('bar')
    r = self.run_bzr('find-merge-base foo bar')[0]
    self.assertEqual('merge base is revision null:\n', r)