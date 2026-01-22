from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_unversion(self):
    """Some test for unversion of a memory tree."""
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    with tree.lock_write():
        tree.add(['', 'foo'], kinds=['directory', 'file'])
        tree.unversion(['foo'])
        self.assertFalse(tree.is_versioned('foo'))
        self.assertFalse(tree.has_filename('foo'))