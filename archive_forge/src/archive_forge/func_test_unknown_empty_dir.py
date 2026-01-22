import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_unknown_empty_dir(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    self.build_tree(['tree1/a/', 'tree1/b/', 'tree2/a/', 'tree2/b/'])
    self.build_tree_contents([('tree1/b/file', b'contents\n'), ('tree2/b/file', b'contents\n')])
    tree1.add(['a', 'b', 'b/file'], ids=[b'a-id', b'b-id', b'b-file-id'])
    tree2.add(['a', 'b', 'b/file'], ids=[b'a-id', b'b-id', b'b-file-id'])
    self.build_tree(['tree2/a/file', 'tree2/a/dir/', 'tree2/a/dir/subfile'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_cannot_represent_unversioned(tree2)
    if tree2.has_versioned_directories():
        expected = self.sorted([self.unversioned(tree2, 'a/file'), self.unversioned(tree2, 'a/dir')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))
    else:
        expected = self.sorted([self.unversioned(tree2, 'a/file'), self.unversioned(tree2, 'a/dir/subfile')])
        self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))