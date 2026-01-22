from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_subtree_and_children(self):
    """Passing a child id will raise NoSuchId.

        This is because the parent directory will have already been removed.
        """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'a/c', 'd'])
    tree.add(['a', 'a/b', 'a/c', 'd'])
    with tree.lock_write():
        tree.unversion(['a/b', 'a'])
        self.assertFalse(tree.is_versioned('a'))
        self.assertFalse(tree.is_versioned('a/b'))
        self.assertFalse(tree.is_versioned('a/c'))
        self.assertTrue(tree.is_versioned('d'))
        self.assertTrue(tree.has_filename('a'))
        self.assertTrue(tree.has_filename('a/b'))
        self.assertTrue(tree.has_filename('a/c'))
        self.assertTrue(tree.has_filename('d'))