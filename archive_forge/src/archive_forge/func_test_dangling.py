import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_dangling(self):
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['2/a'])
    tree2.add('a')
    os.unlink('2/a')
    self.build_tree(['1/b'])
    tree1.add('b')
    os.unlink('1/b')
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    d = self.intertree_class(tree1, tree2).compare()
    self.assertEqual([], d.added)
    self.assertEqual([], d.modified)
    self.assertEqual([], d.removed)
    self.assertEqual([], d.renamed)
    self.assertEqual([], d.unchanged)