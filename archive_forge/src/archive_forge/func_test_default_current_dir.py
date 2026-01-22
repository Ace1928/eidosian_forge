from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport
def test_default_current_dir(self):
    self.run_bzr('init-shared-repo a')
    out, err = self.run_bzr('branches', working_dir='a')
    self.assertEqual(out, '')