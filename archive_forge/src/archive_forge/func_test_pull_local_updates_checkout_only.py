from breezy import branch, controldir, errors, revision
from breezy.tests import TestNotApplicable, fixtures, per_branch
def test_pull_local_updates_checkout_only(self):
    """Pulling --local into a checkout updates the checkout and not the
        master branch"""
    master_tree = self.make_branch_and_tree('master')
    rev1 = master_tree.commit('master')
    checkout = master_tree.branch.create_checkout('checkout')
    other = master_tree.branch.controldir.sprout('other').open_workingtree()
    rev2 = other.commit('other commit')
    checkout.branch.pull(other.branch, local=True)
    self.assertEqual(rev2, checkout.branch.last_revision())
    self.assertEqual(rev1, master_tree.branch.last_revision())