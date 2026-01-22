import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_no_root(self):
    tree = self.make_branch_and_tree('.')
    self.run_bzr_error(['brz: ERROR: can not move root of branch'], 'mv . a')