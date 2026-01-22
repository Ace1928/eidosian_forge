import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_trees_with_deleted_dir(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree1/a', 'tree1/b/', 'tree1/b/c', 'tree1/b/d/', 'tree1/b/d/e', 'tree1/f/', 'tree1/f/g', 'tree2/a', 'tree2/f/', 'tree2/f/g'])
    tree1.add(['a', 'b', 'b/c', 'b/d/', 'b/d/e', 'f', 'f/g'], ids=[b'a-id', b'b-id', b'c-id', b'd-id', b'e-id', b'f-id', b'g-id'])
    tree2.add(['a', 'f', 'f/g'], ids=[b'a-id', b'f-id', b'g-id'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    expected = [self.changed_content(tree2, 'a'), self.changed_content(tree2, 'f/g'), self.deleted(tree1, 'b'), self.deleted(tree1, 'b/c'), self.deleted(tree1, 'b/d'), self.deleted(tree1, 'b/d/e')]
    self.assertEqualIterChanges(expected, self.do_iter_changes(tree1, tree2))
    self.check_has_changes(True, tree1, tree2)