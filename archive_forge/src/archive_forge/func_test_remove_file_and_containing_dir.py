from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_file_and_containing_dir(self):
    tree = self.get_committed_tree(['config/', 'config/file'])
    tree.remove('config/file', keep_files=False)
    tree.remove('config', keep_files=False)
    self.assertPathDoesNotExist('config/file')
    self.assertPathDoesNotExist('config')
    tree._validate()