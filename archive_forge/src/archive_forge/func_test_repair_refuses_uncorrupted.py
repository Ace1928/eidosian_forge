from breezy import workingtree
from breezy.tests import TestCaseWithTransport
def test_repair_refuses_uncorrupted(self):
    tree = self.make_initial_tree()
    self.run_bzr_error(['The tree does not appear to be corrupt', '"brz revert"', '--force'], 'repair-workingtree -d tree')