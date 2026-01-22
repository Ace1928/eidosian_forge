import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_unversioned_files(self):
    self.build_tree(files)
    tree = self.make_branch_and_tree('.')
    self.run_bzr_remove_changed_files(files)