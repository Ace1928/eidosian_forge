from breezy import tests, workingtree
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack4
from breezy.bzr.knitrepo import RepositoryFormatKnit4
def test_split_tree_failure(self):
    tree = self.make_branch_and_tree('tree', format='pack-0.92')
    self.build_tree(['tree/subtree/'])
    tree.add('subtree')
    tree.commit('added subtree')
    self.run_bzr_error(('must upgrade your branch at .*tree', 'rich roots'), 'split tree/subtree')