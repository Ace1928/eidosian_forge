from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_standalone_to_standalone(self):
    tree = self.make_branch_and_tree('tree')
    self.assertRaises(reconfigure.AlreadyStandalone, reconfigure.Reconfigure.to_standalone, tree.controldir)