from breezy import errors, tests
from breezy import transport as _mod_transport
from breezy.tests import per_workingtree
def test_get_zeroth_basis_tree_via_revision_tree(self):
    tree = self.make_branch_and_tree('.')
    try:
        revision_tree = tree.revision_tree(tree.last_revision())
    except errors.NoSuchRevision:
        return
    basis_tree = tree.basis_tree()
    self.assertTreesEqual(revision_tree, basis_tree)