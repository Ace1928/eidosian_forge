import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_trees_with_special_names(self):
    tree1, tree2, paths = self.make_trees_with_special_names()
    expected = self.sorted((self.changed_content(tree2, p) for p in paths if p.endswith('/f')))
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
    self.check_has_changes(True, tree1, tree2)