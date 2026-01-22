import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def make_tree_with_special_names(self):
    """Create a tree with filenames chosen to exercise the walk order."""
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    paths = self._create_special_names(tree2, 'tree2')
    tree2.commit('initial', rev_id=b'rev-1')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    return (tree1, tree2, paths)