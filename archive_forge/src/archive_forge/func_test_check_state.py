from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_check_state(self):
    tree = self.make_branch_and_tree('tree')
    tree.check_state()