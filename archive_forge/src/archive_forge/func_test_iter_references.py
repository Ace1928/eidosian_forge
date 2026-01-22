from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_iter_references(self):
    tree, subtree = self.create_nested()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(['subtree'], list(tree.iter_references()))