import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_with_existing_backup_dir(self):
    self.make_branch_and_tree('old_format_branch', format='knit')
    t = self.get_transport('old_format_branch')
    url = t.base
    display_url = t.local_abspath('.')
    backup_dir1 = 'backup.bzr.~1~'
    backup_dir2 = 'backup.bzr.~2~'
    t.mkdir(backup_dir1)
    out, err = self.run_bzr(['upgrade', '--format=2a', url])
    self.assertEqualDiff('Upgrading branch {}/ ...\nstarting upgrade of {}/\nmaking backup of {}/.bzr\n  to {}/{}\nstarting repository conversion\nrepository converted\nfinished\n'.format(display_url, display_url, display_url, display_url, backup_dir2), out)
    self.assertEqualDiff('', err)
    self.assertTrue(isinstance(controldir.ControlDir.open(self.get_url('old_format_branch'))._format, bzrdir.BzrDirMetaFormat1))
    self.assertTrue(t.has(backup_dir2))