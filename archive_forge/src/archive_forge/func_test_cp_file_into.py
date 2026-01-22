import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_cp_file_into(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['sub1/', 'sub1/hello.txt', 'sub2/'])
    tree.add(['sub1', 'sub1/hello.txt', 'sub2'])
    self.run_bzr('cp sub1/hello.txt sub2')
    self.assertInWorkingTree('sub1')
    self.assertInWorkingTree('sub1/hello.txt')
    self.assertInWorkingTree('sub2')
    self.assertInWorkingTree('sub2/hello.txt')