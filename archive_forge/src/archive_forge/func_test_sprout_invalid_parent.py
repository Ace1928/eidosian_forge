from breezy import branch, errors
from breezy.tests import per_branch, test_server
def test_sprout_invalid_parent(self):
    branch_b = self.get_branch_with_invalid_parent()
    branch_c = branch_b.controldir.sprout('c').open_branch()
    self.assertEqual(branch_b.base, branch_c.get_parent())