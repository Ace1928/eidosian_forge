from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_no_tree(self):
    tree = self.example_tree('a')
    tree.controldir.destroy_workingtree()
    self.run_bzr('rmbranch', working_dir='a')
    dir = controldir.ControlDir.open('a')
    self.assertFalse(dir.has_branch())