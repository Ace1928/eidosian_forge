from breezy import branch, controldir, errors, revision
from breezy.tests import TestNotApplicable, fixtures, per_branch
def test_pull_local_raises_LocalRequiresBoundBranch_on_unbound(self):
    """Pulling --local into a branch that is not bound should fail."""
    master_tree = self.make_branch_and_tree('branch')
    rev1 = master_tree.commit('master')
    other = master_tree.branch.controldir.sprout('other').open_workingtree()
    rev2 = other.commit('other commit')
    self.assertRaises(errors.LocalRequiresBoundBranch, master_tree.branch.pull, other.branch, local=True)
    self.assertEqual(rev1, master_tree.branch.last_revision())