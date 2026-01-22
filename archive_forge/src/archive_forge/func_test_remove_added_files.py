from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_added_files(self):
    """Removal of newly added files must back them up."""
    tree = self.get_tree(TestRemove.files)
    tree.add(TestRemove.files)
    tree.remove(TestRemove.files, keep_files=False)
    self.assertNotInWorkingTree(TestRemove.files)
    if tree.has_versioned_directories():
        self.assertPathExists(TestRemove.backup_files)
    else:
        self.assertPathExists(TestRemove.backup_files_no_version_dirs)
    tree._validate()