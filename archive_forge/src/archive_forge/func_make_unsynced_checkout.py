from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def make_unsynced_checkout(self):
    parent = self.make_branch('parent')
    checkout = parent.create_checkout('checkout')
    checkout.commit('test', rev_id=b'new-commit', local=True)
    reconfiguration = reconfigure.Reconfigure.to_lightweight_checkout(checkout.controldir)
    return (checkout, parent, reconfiguration)