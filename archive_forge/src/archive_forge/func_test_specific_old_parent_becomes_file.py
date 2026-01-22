import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_specific_old_parent_becomes_file(self):
    tree1 = self.make_branch_and_tree('1')
    tree1.mkdir('a', b'a-old-id')
    tree1.mkdir('a/reparented', b'reparented-id')
    tree1.mkdir('a/deleted', b'deleted-id')
    tree2 = self.make_to_branch_and_tree('2')
    tree2.set_root_id(tree1.path2id(''))
    tree2.mkdir('a', b'a-new-id')
    tree2.mkdir('a/reparented', b'reparented-id')
    tree2.add(['b'], ['file'], [b'a-old-id'])
    tree2.put_file_bytes_non_atomic('b', b'')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqualIterChanges([self.kind_changed(tree1, tree2, 'a', 'b'), self.added(tree2, 'a'), self.renamed(tree1, tree2, 'a/reparented', 'a/reparented', False), self.deleted(tree1, 'a/deleted')], self.do_iter_changes(tree1, tree2, specific_files=['a/reparented']))