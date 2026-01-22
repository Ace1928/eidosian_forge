import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_dirs(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['hello.txt', 'sub1/'])
    tree.add(['hello.txt', 'sub1'])
    self.run_bzr('mv sub1 sub2')
    self.assertMoved('sub1', 'sub2')
    self.run_bzr('mv hello.txt sub2')
    self.assertMoved('hello.txt', 'sub2/hello.txt')
    self.build_tree(['sub1/'])
    tree.add(['sub1'])
    self.run_bzr('mv sub2/hello.txt sub1')
    self.assertMoved('sub2/hello.txt', 'sub1/hello.txt')
    self.run_bzr('mv sub2 sub1')
    self.assertMoved('sub2', 'sub1/sub2')