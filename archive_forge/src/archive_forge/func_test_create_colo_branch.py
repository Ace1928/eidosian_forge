from breezy import controldir, errors, tests
from breezy.tests import per_controldir
def test_create_colo_branch(self):
    made_control = self.make_controldir_with_repo()
    self.assertRaises(controldir.NoColocatedBranchSupport, made_control.create_branch, 'colo')