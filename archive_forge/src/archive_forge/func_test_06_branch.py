import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_06_branch(self):
    """branch b1=>b2 should preserve the executable bits"""
    wt2, r1 = self.commit_and_branch()
    self.check_exist(wt2)