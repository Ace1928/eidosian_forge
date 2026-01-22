import os
import breezy.osutils
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_revert_newly_added(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file'])
    tree.add(['file'])
    out, err = self.run_bzr('revert')
    self.assertEqual('', out)
    self.assertEqual('-   file\n', err)