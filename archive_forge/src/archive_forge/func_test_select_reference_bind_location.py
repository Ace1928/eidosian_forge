from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_select_reference_bind_location(self):
    branch = self.make_branch('branch')
    checkout = branch.create_checkout('checkout', lightweight=True)
    reconfiguration = reconfigure.Reconfigure(checkout.controldir)
    self.assertEqual(branch.base, reconfiguration._select_bind_location())