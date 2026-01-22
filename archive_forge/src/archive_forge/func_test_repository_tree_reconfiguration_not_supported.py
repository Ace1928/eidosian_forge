from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_repository_tree_reconfiguration_not_supported(self):
    tree = self.make_branch_and_tree('tree')
    e = self.assertRaises(reconfigure.ReconfigurationNotSupported, reconfigure.Reconfigure.set_repository_trees, tree.controldir, None)
    self.assertContainsRe(str(e), "Requested reconfiguration of '.*' is not supported.")