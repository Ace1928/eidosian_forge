import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_from_any_repo_revision(self):
    """We should be able to sprout from any revision."""
    wt = self.make_branch_and_tree('source')
    self.build_tree(['source/a'])
    wt.add('a')
    rev1a = wt.commit('rev1a')
    wt.branch.set_last_revision_info(0, _mod_revision.NULL_REVISION)
    wt.set_last_revision(_mod_revision.NULL_REVISION)
    wt.revert()
    wt.commit('rev1b')
    wt2 = wt.controldir.sprout('target', revision_id=rev1a).open_workingtree()
    self.assertEqual(rev1a, wt2.last_revision())
    self.assertPathExists('target/a')