import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_05_removed_and_committed(self):
    """Check that reverting to an earlier commit restores them"""
    r1 = self.wt.commit('adding a,b')
    os.remove('b1/a')
    os.remove('b1/b')
    r2 = self.wt.commit('removed')
    self.check_empty(self.wt)
    rev_tree = self.wt.branch.repository.revision_tree(r1)
    self.wt.revert(old_tree=rev_tree, backups=False)
    self.check_exist(self.wt)