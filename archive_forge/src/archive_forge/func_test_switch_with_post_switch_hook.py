import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_with_post_switch_hook(self):
    from breezy import branch as _mod_branch
    calls = []
    _mod_branch.Branch.hooks.install_named_hook('post_switch', calls.append, None)
    self.make_branch_and_tree('branch')
    self.run_bzr('branch branch branch2')
    self.run_bzr('checkout branch checkout')
    os.chdir('checkout')
    self.assertLength(0, calls)
    out, err = self.run_bzr('switch ../branch2')
    self.assertLength(1, calls)