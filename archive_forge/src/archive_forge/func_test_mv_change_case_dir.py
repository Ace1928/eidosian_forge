import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_change_case_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo/'])
    tree.add(['foo'])
    self.run_bzr('mv foo Foo')
    shape = sorted(os.listdir('.'))
    self.assertEqual(['.bzr', 'Foo'], shape)
    self.assertInWorkingTree('Foo')
    self.assertNotInWorkingTree('foo')