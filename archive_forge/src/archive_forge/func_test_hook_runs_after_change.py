from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def test_hook_runs_after_change(self):
    """The hook runs *after* the branch's last_revision_info has changed.
        """
    branch = self.make_branch_with_revision_ids(b'revid-one')

    def assertBranchAtRevision1(params):
        self.assertEqual((0, revision.NULL_REVISION), params.branch.last_revision_info())
    _mod_branch.Branch.hooks.install_named_hook('post_change_branch_tip', assertBranchAtRevision1, None)
    branch.set_last_revision_info(0, revision.NULL_REVISION)