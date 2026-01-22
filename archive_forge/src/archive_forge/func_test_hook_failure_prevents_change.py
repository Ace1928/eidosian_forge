from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def test_hook_failure_prevents_change(self):
    """If a hook raises an exception, the change does not take effect."""
    branch = self.make_branch_with_revision_ids(b'one-\xc2\xb5', b'two-\xc2\xb5')

    class PearShapedError(Exception):
        pass

    def hook_that_raises(params):
        raise PearShapedError()
    _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', hook_that_raises, None)
    hook_failed_exc = self.assertRaises(PearShapedError, branch.set_last_revision_info, 0, revision.NULL_REVISION)
    self.assertEqual((2, b'two-\xc2\xb5'), branch.last_revision_info())