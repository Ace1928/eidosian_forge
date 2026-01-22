import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_cp_file(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['hello.txt'])
    tree.add(['hello.txt'])
    self.run_bzr('cp hello.txt hallo.txt')
    self.assertInWorkingTree('hello.txt')
    self.assertInWorkingTree('hallo.txt')