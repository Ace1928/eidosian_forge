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
def test_no_metadir_support(self):
    url = self.get_url()
    bdir = bzrdir.BzrDirMetaFormat1().initialize(url)
    bdir.create_repository()
    self.assertRaises(errors.IncompatibleFormat, BzrBranchFormat4().initialize, bdir)