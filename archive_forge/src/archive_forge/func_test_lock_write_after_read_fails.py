from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_lock_write_after_read_fails(self):
    """Check that we error when trying to upgrade a read lock to write."""
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    tree.lock_read()
    self.assertRaises(errors.ReadOnlyError, tree.lock_write)
    tree.unlock()