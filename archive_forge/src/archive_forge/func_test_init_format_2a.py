import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_format_2a(self):
    """Smoke test for constructing a format 2a repository."""
    out, err = self.run_bzr('init --format=2a')
    self.assertEqual('Created a standalone tree (format: 2a)\n', out)
    self.assertEqual('', err)