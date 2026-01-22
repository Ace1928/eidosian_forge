from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_check_broken_dirstate(self):
    tree = self.make_tree_with_broken_dirstate('tree')
    self.assertRaises(errors.BzrError, tree.check_state)