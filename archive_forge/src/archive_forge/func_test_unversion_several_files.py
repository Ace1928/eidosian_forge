from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_several_files(self):
    """After unversioning several files, they should not be versioned."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b', 'c'])
    tree.add(['a', 'b', 'c'])
    tree.lock_write()
    self.assertTrue(tree.is_versioned('a'))
    tree.unversion(['a', 'b'])
    self.assertFalse(tree.is_versioned('a'))
    self.assertFalse(tree.is_versioned('a'))
    self.assertFalse(tree.is_versioned('b'))
    self.assertTrue(tree.is_versioned('c'))
    self.assertTrue(tree.has_filename('a'))
    self.assertTrue(tree.has_filename('b'))
    self.assertTrue(tree.has_filename('c'))
    tree.unlock()
    tree = tree.controldir.open_workingtree()
    self.addCleanup(tree.lock_read().unlock)
    self.assertFalse(tree.is_versioned('a'))
    self.assertFalse(tree.is_versioned('b'))
    self.assertTrue(tree.is_versioned('c'))
    self.assertTrue(tree.has_filename('a'))
    self.assertTrue(tree.has_filename('b'))
    self.assertTrue(tree.has_filename('c'))