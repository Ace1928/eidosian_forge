from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_no_arg(self):
    self.example_tree('a')
    self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch a')
    self.run_bzr('rmbranch --force', working_dir='a')
    dir = controldir.ControlDir.open('a')
    self.assertFalse(dir.has_branch())