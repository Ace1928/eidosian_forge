from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_checkout_to_branch(self):
    branch = self.make_branch('branch')
    checkout = branch.create_checkout('checkout')
    reconfiguration = reconfigure.Reconfigure.to_branch(checkout.controldir)
    reconfiguration.apply()
    reconfigured = controldir.ControlDir.open('checkout').open_branch()
    self.assertIs(None, reconfigured.get_bound_location())