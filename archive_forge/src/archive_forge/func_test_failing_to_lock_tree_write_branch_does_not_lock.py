import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_failing_to_lock_tree_write_branch_does_not_lock(self):
    """If the branch cannot be read locked, dont lock the tree."""
    wt = self.make_branch_and_tree('.')
    branch_copy = branch.Branch.open('.')
    branch_copy.lock_write()
    try:
        try:
            wt.lock_tree_write()
        except errors.LockError:
            self.assertFalse(wt.is_locked())
            self.assertFalse(wt.branch.is_locked())
            return
        else:
            wt.unlock()
    finally:
        branch_copy.unlock()