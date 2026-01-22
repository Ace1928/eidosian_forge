import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_deleted_files(self):
    tree = self._make_tree_and_add(files)
    self.run_bzr("commit -m 'added files'")
    my_files = [f for f in files]
    my_files.sort(reverse=True)
    for f in my_files:
        osutils.delete_any(f)
    self.assertInWorkingTree(files)
    self.assertPathDoesNotExist(files)
    self.run_bzr('remove ' + ' '.join(files))
    self.assertNotInWorkingTree(a)
    self.assertPathDoesNotExist(files)