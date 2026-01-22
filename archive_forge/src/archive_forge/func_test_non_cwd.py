from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_non_cwd(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/dir/', 'tree/dir/file'])
    tree.add(['dir', 'dir/file'])
    tree.commit('add file')
    tree.remove('dir/', keep_files=False)
    self.assertPathDoesNotExist('tree/dir/file')
    self.assertNotInWorkingTree('tree/dir/file', 'tree')
    tree._validate()