from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_requires_write_lock(self):
    """WT.unversion([]) in a read lock raises ReadOnlyError."""
    tree = self.make_branch_and_tree('.')
    tree.lock_read()
    self.assertRaises(errors.ReadOnlyError, tree.unversion, [])
    tree.unlock()