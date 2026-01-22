from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_last_revision(self):
    """There should be a last revision method we can call."""
    tree = self.make_branch_and_memory_tree('branch')
    with tree.lock_write():
        tree.add('')
        rev_id = tree.commit('first post')
    self.assertEqual(rev_id, tree.last_revision())