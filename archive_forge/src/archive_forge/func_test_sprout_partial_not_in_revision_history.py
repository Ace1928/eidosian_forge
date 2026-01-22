import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_partial_not_in_revision_history(self):
    """We should be able to sprout from any revision in ancestry."""
    wt = self.make_branch_and_tree('source')
    self.build_tree(['source/a'])
    wt.add('a')
    rev1 = wt.commit('rev1')
    rev2_alt = wt.commit('rev2-alt')
    wt.set_parent_ids([rev1])
    wt.branch.set_last_revision_info(1, rev1)
    rev2 = wt.commit('rev2')
    wt.set_parent_ids([rev2, rev2_alt])
    wt.commit('rev3')
    repo = self.make_repository('target')
    repo.fetch(wt.branch.repository)
    branch2 = wt.branch.sprout(repo.controldir, revision_id=rev2_alt)
    self.assertEqual((2, rev2_alt), branch2.last_revision_info())
    self.assertEqual(rev2_alt, branch2.last_revision())