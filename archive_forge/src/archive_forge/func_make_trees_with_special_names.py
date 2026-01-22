import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def make_trees_with_special_names(self):
    """Both trees will use the special names.

        But the contents will differ for each file.
        """
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    paths = self._create_special_names(tree1, 'tree1')
    paths = self._create_special_names(tree2, 'tree2')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    return (tree1, tree2, paths)