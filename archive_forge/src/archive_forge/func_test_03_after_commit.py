import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_03_after_commit(self):
    """Commit the change, and check the history"""
    r1 = self.wt.commit('adding a,b')
    rev_tree = self.wt.branch.repository.revision_tree(r1)
    self.check_exist(rev_tree)