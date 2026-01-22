import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_empty_to_abc_content_c_only(self):
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree1 = self.get_tree_no_parents_no_content(tree1)
    tree2 = self.get_tree_no_parents_abc_content(tree2)
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    d = self.intertree_class(tree1, tree2).compare(specific_files=['b/c'])
    self.assertEqual([('b', 'directory'), ('b/c', 'file')], [(c.path[1], c.kind[1]) for c in d.added])
    self.assertEqual([], d.modified)
    self.assertEqual([], d.removed)
    self.assertEqual([], d.renamed)
    self.assertEqual([], d.unchanged)