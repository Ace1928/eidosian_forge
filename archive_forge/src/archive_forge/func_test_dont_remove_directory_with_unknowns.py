from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_dont_remove_directory_with_unknowns(self):
    """Directories with unknowns should be backed up."""
    directories = ['a/', 'b/', 'c/', 'c/c/', 'c/blah']
    tree = self.get_committed_tree(directories)
    self.build_tree(['a/unknown_file'])
    tree.remove('a', keep_files=False)
    self.assertPathExists('a.~1~/unknown_file')
    self.build_tree(['b/unknown_directory'])
    tree.remove('b', keep_files=False)
    self.assertPathExists('b.~1~/unknown_directory')
    self.build_tree(['c/c/unknown_file'])
    tree.remove('c/c', keep_files=False)
    self.assertPathExists('c/c.~1~/unknown_file')
    tree.remove('c', keep_files=False)
    self.assertPathExists('c.~1~/')
    self.assertNotInWorkingTree(directories)
    tree._validate()