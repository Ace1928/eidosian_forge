from breezy import controldir, errors, tests
from breezy.tests import per_controldir
def test_destroy_colocated_branch(self):
    branch = self.make_branch('branch')
    self.assertRaises((controldir.NoColocatedBranchSupport, errors.UnsupportedOperation), branch.controldir.destroy_branch, 'colo')