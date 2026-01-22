import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_create_branch_short_name(self):
    branch = self.make_branch('branch')
    tree = branch.create_checkout('tree', lightweight=True)
    tree.commit('one', rev_id=b'rev-1')
    self.run_bzr('switch -b branch2', working_dir='tree')
    tree = WorkingTree.open('tree')
    self.assertEqual(branch.base[:-1] + '2/', tree.branch.base)