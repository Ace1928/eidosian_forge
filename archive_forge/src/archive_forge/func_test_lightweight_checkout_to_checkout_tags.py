from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_lightweight_checkout_to_checkout_tags(self):
    reconfiguration, checkout = self.prepare_lightweight_checkout_to_checkout()
    checkout.branch.tags.set_tag('foo', b'bar')
    reconfiguration.apply()
    checkout_branch = checkout.controldir.open_branch()
    self.assertEqual(b'bar', checkout_branch.tags.lookup_tag('foo'))