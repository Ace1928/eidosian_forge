import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def test_iter_entries_with_missing_reference(self):
    tree, subtree = self.create_nested()
    shutil.rmtree('wt/subtree')
    expected = [('', 'directory'), ('subtree', 'tree-reference')]
    with tree.lock_read():
        self.assertRaises(MissingNestedTree, list, tree.iter_entries_by_dir(recurse_nested=True))