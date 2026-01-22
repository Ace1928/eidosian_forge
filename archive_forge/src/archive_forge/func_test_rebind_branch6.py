from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_rebind_branch6(self):
    self.setup_rebind('dirstate-tags')
    self.run_bzr('bind', working_dir='branch2')
    b = branch.Branch.open('branch2')
    self.assertEndsWith(b.get_bound_location(), '/branch1/')