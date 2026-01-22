import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_new_no_files_specified(self):
    tree = self.make_branch_and_tree('.')
    self.run_bzr_error(['brz: ERROR: No matching files.'], 'remove --new')
    self.run_bzr_error(['brz: ERROR: No matching files.'], 'remove --new .')