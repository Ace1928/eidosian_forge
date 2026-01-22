import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_append_revisions_only(self):
    self.run_bzr('init --format=dirstate-tags normal_branch6')
    branch = _mod_branch.Branch.open('normal_branch6')
    self.assertEqual(None, branch.get_append_revisions_only())
    self.run_bzr('init --append-revisions-only --format=dirstate-tags branch6')
    branch = _mod_branch.Branch.open('branch6')
    self.assertEqual(True, branch.get_append_revisions_only())
    self.run_bzr_error(['cannot be set to append-revisions-only'], 'init --append-revisions-only --format=knit knit')