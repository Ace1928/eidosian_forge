import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_control_dir(self):
    old_format = OldBzrDirFormat()
    self.addCleanup(bzr.BzrProber.formats.remove, old_format.get_format_string())
    bzr.BzrProber.formats.register(old_format.get_format_string(), old_format)
    self.addCleanup(controldir.ControlDirFormat._set_default_format, controldir.ControlDirFormat.get_default_format())
    path = 'old_format_branch'
    self.make_branch_and_tree(path, format=old_format)
    transport = self.get_transport(path)
    url = transport.base
    display_url = transport.local_abspath('.')
    controldir.ControlDirFormat._set_default_format(old_format)
    backup_dir = 'backup.bzr.~1~'
    out, err = self.run_bzr(['upgrade', '--format=2a', url])
    self.assertEqualDiff('Upgrading branch {}/ ...\nstarting upgrade of {}/\nmaking backup of {}/.bzr\n  to {}/{}\nstarting upgrade from old test format to 2a\nfinished\n'.format(display_url, display_url, display_url, display_url, backup_dir), out)
    self.assertEqualDiff('', err)
    self.assertTrue(isinstance(controldir.ControlDir.open(self.get_url(path))._format, bzrdir.BzrDirMetaFormat1))