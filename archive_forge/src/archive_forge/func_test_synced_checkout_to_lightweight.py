from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_synced_checkout_to_lightweight(self):
    checkout, parent, reconfiguration = self.make_unsynced_checkout()
    parent.pull(checkout.branch)
    reconfiguration.apply()
    wt = checkout.controldir.open_workingtree()
    self.assertTrue(parent.repository.has_same_location(wt.branch.repository))
    parent.repository.get_revision(b'new-commit')
    self.assertRaises(errors.NoRepositoryPresent, checkout.controldir.open_repository)