from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_use_shared_to_use_shared(self):
    tree = self.make_repository_tree()
    self.assertRaises(reconfigure.AlreadyUsingShared, reconfigure.Reconfigure.to_use_shared, tree.controldir)