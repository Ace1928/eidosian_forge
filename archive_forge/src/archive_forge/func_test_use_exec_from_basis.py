import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_use_exec_from_basis(self):
    self.wt._supports_executable = lambda: False
    self.addCleanup(self.wt.lock_read().unlock)
    self.assertTrue(self.wt.is_executable('a'))
    self.assertFalse(self.wt.is_executable('b'))