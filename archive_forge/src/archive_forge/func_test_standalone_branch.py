from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport
def test_standalone_branch(self):
    self.make_branch('a')
    out, err = self.run_bzr('branches a')
    self.assertEqual(out, '* (default)\n')