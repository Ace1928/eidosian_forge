import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_specific_content_modification_grabs_parents(self):
    tree1 = self.make_branch_and_tree('1')
    tree1.mkdir('changing', b'parent-id')
    tree1.mkdir('changing/unchanging', b'mid-id')
    tree1.add(['changing/unchanging/file'], ['file'], [b'file-id'])
    tree1.put_file_bytes_non_atomic('changing/unchanging/file', b'a file')
    tree2 = self.make_to_branch_and_tree('2')
    tree2.set_root_id(tree1.path2id(''))
    tree2.mkdir('changed', b'parent-id')
    tree2.mkdir('changed/unchanging', b'mid-id')
    tree2.add(['changed/unchanging/file'], ['file'], [b'file-id'])
    tree2.put_file_bytes_non_atomic('changed/unchanging/file', b'changed content')
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    root_id = tree1.path2id('')
    self.assertEqualIterChanges([self.renamed(tree1, tree2, 'changing', 'changed', False), self.renamed(tree1, tree2, 'changing/unchanging/file', 'changed/unchanging/file', True)], self.do_iter_changes(tree1, tree2, specific_files=['changed/unchanging/file']))