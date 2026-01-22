import os
from breezy.branch import Branch
from breezy.osutils import pathjoin
from breezy.tests import TestCaseInTempDir, TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_mkdir_in_subdir(self):
    """'brz mkdir' operation in subdirectory"""
    self.make_branch_and_tree('.')
    self.run_bzr(['mkdir', 'dir'])
    self.assertTrue(os.path.isdir('dir'))
    self.log('Run mkdir in subdir')
    self.run_bzr(['mkdir', 'subdir'], working_dir='dir')
    self.assertTrue(os.path.isdir('dir/subdir'))
    wt = WorkingTree.open('.')
    delta = wt.changes_from(wt.basis_tree())
    self.log('delta.added = %r' % delta.added)
    self.assertEqual(len(delta.added), 2)
    self.assertEqual(delta.added[0].path[1], 'dir')
    self.assertEqual(delta.added[1].path[1], pathjoin('dir', 'subdir'))
    self.assertFalse(delta.modified)