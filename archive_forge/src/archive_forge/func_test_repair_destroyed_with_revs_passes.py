from breezy import workingtree
from breezy.tests import TestCaseWithTransport
def test_repair_destroyed_with_revs_passes(self):
    tree = self.make_initial_tree()
    self.break_dirstate(tree, completely=True)
    self.run_bzr('repair-workingtree -d tree -r -1')
    tree = workingtree.WorkingTree.open('tree')
    tree.check_state()