import os
import stat
import sys
from breezy import tests
from breezy.bzr.branch import BzrBranch
from breezy.bzr.remote import RemoteBranchFormat
from breezy.controldir import ControlDir
from breezy.tests.test_permissions import check_mode_r
def test_new_branch_group_sticky_bit(self):
    if isinstance(self.branch_format, RemoteBranchFormat):
        raise tests.TestNotApplicable('Remote branches have no permission logic')
    if sys.platform == 'win32':
        raise tests.TestNotApplicable('chmod has no effect on win32')
    elif sys.platform == 'darwin' or 'freebsd' in sys.platform:
        os.chown(self.test_dir, os.getuid(), os.getgid())
    t = self.make_branch_and_tree('.')
    b = t.branch
    if not isinstance(b, BzrBranch):
        raise tests.TestNotApplicable('Only applicable to bzr branches')
    os.mkdir('b')
    os.chmod('b', 1535)
    b = self.make_branch('b')
    self.assertEqualMode(1535, b.controldir._get_dir_mode())
    self.assertEqualMode(438, b.controldir._get_file_mode())
    self.assertEqualMode(1535, b.control_files._dir_mode)
    self.assertEqualMode(438, b.control_files._file_mode)
    check_mode_r(self, 'b/.bzr', 438, 1535)
    os.mkdir('c')
    os.chmod('c', 1512)
    b = self.make_branch('c')
    self.assertEqualMode(1512, b.controldir._get_dir_mode())
    self.assertEqualMode(416, b.controldir._get_file_mode())
    self.assertEqualMode(1512, b.control_files._dir_mode)
    self.assertEqualMode(416, b.control_files._file_mode)
    check_mode_r(self, 'c/.bzr', 416, 1512)