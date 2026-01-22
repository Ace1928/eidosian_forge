import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def test_iter_entries_with_followed_reference(self):
    tree, subtree = self.create_nested()
    expected = [('', 'directory'), ('subtree', 'directory'), ('subtree/a', 'file')]
    with tree.lock_read():
        path_entries = list(tree.iter_entries_by_dir(recurse_nested=True))
        actual = [(path, ie.kind) for path, ie in path_entries]
    self.assertEqual(expected, actual)