import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_modes(self):
    """Test two modes of operation for mv"""
    tree = self.make_branch_and_tree('.')
    files = self.build_tree(['a', 'c', 'subdir/'])
    tree.add(['a', 'c', 'subdir'])
    self.run_bzr('mv a b')
    self.assertMoved('a', 'b')
    self.run_bzr('mv b subdir')
    self.assertMoved('b', 'subdir/b')
    self.run_bzr('mv subdir/b a')
    self.assertMoved('subdir/b', 'a')
    self.run_bzr('mv a c subdir')
    self.assertMoved('a', 'subdir/a')
    self.assertMoved('c', 'subdir/c')
    self.run_bzr('mv subdir/a subdir/newa')
    self.assertMoved('subdir/a', 'subdir/newa')