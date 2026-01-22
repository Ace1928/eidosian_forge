import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_cp_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['hello.txt', 'sub1/'])
    tree.add(['hello.txt', 'sub1'])
    self.run_bzr_error(['^brz: ERROR: Could not copy sub1 => sub2 . sub1 is a directory\\.$'], 'cp sub1 sub2')