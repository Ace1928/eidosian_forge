from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_create_on_branch(self):
    """Creating a mutable tree on a trivial branch works."""
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    self.assertEqual(branch.controldir, tree.controldir)
    self.assertEqual(branch, tree.branch)
    self.assertEqual([], tree.get_parent_ids())