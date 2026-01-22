from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_unchanged_directory(self):
    """Unchanged directories should be deleted."""
    files = ['b/', 'b/c', 'b/sub_directory/', 'b/sub_directory/with_file']
    tree = self.get_committed_tree(files)
    tree.remove('b', keep_files=False)
    self.assertRemovedAndDeleted('b')
    tree._validate()