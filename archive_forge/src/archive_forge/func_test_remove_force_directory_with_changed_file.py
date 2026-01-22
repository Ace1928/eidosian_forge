from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_force_directory_with_changed_file(self):
    """Delete directories with changed files when forced."""
    files = ['b/', 'b/c']
    tree = self.get_committed_tree(files)
    self.build_tree_contents([('b/c', b'some other new content!')])
    tree.remove('b', keep_files=False, force=True)
    self.assertRemovedAndDeleted(files)
    tree._validate()