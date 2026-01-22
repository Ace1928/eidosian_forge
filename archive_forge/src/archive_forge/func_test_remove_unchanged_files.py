from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_unchanged_files(self):
    """Check that unchanged files are removed and deleted."""
    tree = self.get_committed_tree(TestRemove.files)
    tree.remove(TestRemove.files, keep_files=False)
    self.assertRemovedAndDeleted(TestRemove.files)
    tree._validate()