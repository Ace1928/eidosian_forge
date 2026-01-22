from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_changed_file(self):
    """Removal of changed files must back it up."""
    tree = self.get_committed_tree(['a'])
    self.build_tree_contents([('a', b'some other new content!')])
    self.assertInWorkingTree('a')
    tree.remove('a', keep_files=False)
    self.assertNotInWorkingTree(TestRemove.files)
    self.assertPathExists('a.~1~')
    tree._validate()