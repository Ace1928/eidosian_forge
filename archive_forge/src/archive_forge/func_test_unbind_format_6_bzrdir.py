import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
def test_unbind_format_6_bzrdir(self):
    out, err = self.run_bzr('unbind', retcode=3)
    self.assertEqual('', out)
    cwd = urlutils.local_path_to_url(getcwd())
    self.assertEqual('brz: ERROR: To use this feature you must upgrade your branch at %s/.\n' % cwd, err)