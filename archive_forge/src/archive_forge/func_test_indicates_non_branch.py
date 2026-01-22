from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport
def test_indicates_non_branch(self):
    t = self.make_branch_and_tree('a', format='development-colo')
    t.controldir.create_branch(name='another')
    t.controldir.create_branch(name='colocated')
    out, err = self.run_bzr('branches a')
    self.assertEqual(out, '* (default)\n  another\n  colocated\n')