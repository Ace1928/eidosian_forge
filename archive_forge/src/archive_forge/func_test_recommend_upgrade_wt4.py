import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_recommend_upgrade_wt4(self):
    self.run_bzr('init --format=knit a')
    out, err = self.run_bzr('status a')
    self.assertContainsRe(err, 'brz upgrade .*[/\\\\]a')