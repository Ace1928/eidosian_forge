import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_08_no_op_revert(self):
    """Just do a simple revert without anything changed

        The bits shouldn't swap.
        """
    r1 = self.wt.commit('adding a,b')
    rev_tree = self.wt.branch.repository.revision_tree(r1)
    self.wt.revert(old_tree=rev_tree, backups=False)
    self.check_exist(self.wt)