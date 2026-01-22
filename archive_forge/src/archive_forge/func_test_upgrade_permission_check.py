import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_permission_check(self):
    """'backup.bzr' should retain permissions of .bzr. Bug #262450"""
    self.requireFeature(features.posix_permissions_feature)
    old_perms = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    backup_dir = 'backup.bzr.~1~'
    self.run_bzr('init --format=1.6')
    os.chmod('.bzr', old_perms)
    self.run_bzr('upgrade')
    new_perms = os.stat(backup_dir).st_mode & 511
    self.assertTrue(new_perms == old_perms)