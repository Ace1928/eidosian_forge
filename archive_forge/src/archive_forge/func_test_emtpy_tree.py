from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_emtpy_tree(self):
    """A working tree with no parents."""
    tree = self.make_branch_and_tree('tree')
    basis_tree = tree.basis_tree()
    with basis_tree.lock_read():
        self.assertEqual([], list(basis_tree.list_files(include_root=True)))