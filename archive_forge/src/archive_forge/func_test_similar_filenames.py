import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_similar_filenames(self):
    """Test when we have a few files with similar names."""
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree1/a/', 'tree1/a/b/', 'tree1/a/b/c/', 'tree1/a/b/c/d/', 'tree1/a-c/', 'tree1/a-c/e/', 'tree2/a/', 'tree2/a/b/', 'tree2/a/b/c/', 'tree2/a/b/c/d/', 'tree2/a-c/', 'tree2/a-c/e/'])
    tree1.add(['a', 'a/b', 'a/b/c', 'a/b/c/d', 'a-c', 'a-c/e'], ids=[b'a-id', b'b-id', b'c-id', b'd-id', b'a-c-id', b'e-id'])
    tree2.add(['a', 'a/b', 'a/b/c', 'a/b/c/d', 'a-c', 'a-c/e'], ids=[b'a-id', b'b-id', b'c-id', b'd-id', b'a-c-id', b'e-id'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_cannot_represent_unversioned(tree2)
    self.assertEqual([], self.do_iter_changes(tree1, tree2, want_unversioned=True))
    expected = self.sorted([self.unchanged(tree2, ''), self.unchanged(tree2, 'a'), self.unchanged(tree2, 'a/b'), self.unchanged(tree2, 'a/b/c'), self.unchanged(tree2, 'a/b/c/d'), self.unchanged(tree2, 'a-c'), self.unchanged(tree2, 'a-c/e')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True, include_unchanged=True))