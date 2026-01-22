import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_up_to_date(self):
    self.make_current_format_branch_and_checkout()
    burl = self.get_transport('current_format_branch').local_abspath('.')
    out, err = self.run_bzr('upgrade current_format_branch', retcode=0)
    self.assertEqual('Upgrading branch %s/ ...\nThe branch format %s is already at the most recent format.\n' % (burl, 'Meta directory format 1'), out)