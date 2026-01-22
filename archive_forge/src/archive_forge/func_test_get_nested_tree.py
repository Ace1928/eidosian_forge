from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_get_nested_tree(self):
    tree, subtree = self.create_nested()
    try:
        changes = subtree.changes_from(tree.get_nested_tree('subtree'))
        self.assertFalse(changes.has_changed())
    except MissingNestedTree:
        pass