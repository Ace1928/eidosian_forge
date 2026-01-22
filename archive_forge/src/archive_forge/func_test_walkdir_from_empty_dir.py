import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_walkdir_from_empty_dir(self):
    """Doing a walkdir when the requested prefix is empty dir."""
    self._test_walkdir(self.added, 'added empty dir')