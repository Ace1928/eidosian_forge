from breezy import errors, tests
from breezy import transport as _mod_transport
from breezy.tests import per_workingtree
def test_get_nonzeroth_basis_tree_via_revision_tree(self):
    tree = self.make_branch_and_tree('.')
    revision1 = tree.commit('first post')
    revision_tree = tree.revision_tree(revision1)
    basis_tree = tree.basis_tree()
    self.assertTreesEqual(revision_tree, basis_tree)