from breezy import errors, reconcile
from breezy.bzr.branch import BzrBranch
from breezy.symbol_versioning import deprecated_in
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
def test_reconcile_handles_ghosts_in_revhistory(self):
    tree = self.make_branch_and_tree('test')
    if not tree.branch.repository._format.supports_ghosts:
        raise TestNotApplicable('repository format does not support ghosts')
    tree.set_parent_ids([b'spooky'], allow_leftmost_as_ghost=True)
    r1 = tree.commit('one')
    r2 = tree.commit('two')
    tree.branch.set_last_revision_info(2, r2)
    reconciler = tree.branch.reconcile()
    self.assertEqual(r2, tree.branch.last_revision())