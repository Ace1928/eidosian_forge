from .. import missing, tests
from ..missing import iter_log_revisions
from . import TestCaseWithTransport
def test_same_branch(self):
    tree = self.make_branch_and_tree('tree')
    rev1 = tree.commit('one')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertUnmerged([], [], tree.branch, tree.branch)
    self.assertUnmerged([], [], tree.branch, tree.branch, local_revid_range=(rev1, rev1))