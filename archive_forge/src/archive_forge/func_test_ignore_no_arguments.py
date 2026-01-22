import os
import re
import breezy
from breezy import ignores, osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.osutils import pathjoin
from breezy.tests import TestCaseWithTransport
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_ignore_no_arguments(self):
    """'ignore' with no arguments returns an error"""
    self.make_branch_and_tree('.')
    self.run_bzr_error(('brz: ERROR: ignore requires at least one NAME_PATTERN or --default-rules.\n',), 'ignore')