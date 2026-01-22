import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_specific_old_parent_same_path_new_parent(self):
    tree1 = self.make_branch_and_tree('1')
    tree1.add(['a'], ['file'], [b'a-id'])
    tree1.put_file_bytes_non_atomic('a', b'a file')
    tree2 = self.make_to_branch_and_tree('2')
    tree2.set_root_id(tree1.path2id(''))
    tree2.mkdir('a', b'b-id')
    tree2.add(['a/c'], ['file'], [b'c-id'])
    tree2.put_file_bytes_non_atomic('a/c', b'another file')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqualIterChanges([self.deleted(tree1, 'a'), self.added(tree2, 'a'), self.added(tree2, 'a/c')], self.do_iter_changes(tree1, tree2, specific_files=['a/c']))