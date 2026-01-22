from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_lock_write(self):
    """Check we can lock_write and unlock MemoryTrees."""
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    tree.lock_write()
    tree.unlock()