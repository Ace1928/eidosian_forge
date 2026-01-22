from breezy import workingtree
from breezy.tests import TestCaseWithTransport
def test_repair_naive_destroyed_fails(self):
    tree = self.make_initial_tree()
    self.break_dirstate(tree, completely=True)
    self.run_bzr_error(['the header appears corrupt, try passing'], 'repair-workingtree -d tree')