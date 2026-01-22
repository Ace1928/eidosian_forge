from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_view_command_help(self):
    out, err = self.run_bzr('help view')
    self.assertContainsRe(out, 'Manage filtered views')