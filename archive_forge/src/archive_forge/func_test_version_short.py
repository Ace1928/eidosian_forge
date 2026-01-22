import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
def test_version_short(self):
    self.permit_source_tree_branch_repo()
    out = self.run_bzr(['version', '--short'])[0]
    self.assertEqualDiff(out, breezy.version_string + '\n')