from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_dir_before_bzr(self):
    tree = self.get_committed_tree(['.aaa/', '.aaa/file'])
    tree.remove('.aaa/', keep_files=False)
    self.assertPathDoesNotExist('.aaa/file')
    self.assertPathDoesNotExist('.aaa')
    tree._validate()