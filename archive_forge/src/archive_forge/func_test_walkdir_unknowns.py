import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_walkdir_unknowns(self):
    """unknown files and directories should be reported by walkdirs."""
    self._test_walkdir(self.unknown)