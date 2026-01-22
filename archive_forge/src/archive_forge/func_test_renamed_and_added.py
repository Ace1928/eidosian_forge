import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_renamed_and_added(self):
    """Test when we have renamed a file, and put another in its place."""
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    self.build_tree_contents([('tree1/b', b'b contents\n'), ('tree1/c', b'c contents\n'), ('tree2/a', b'b contents\n'), ('tree2/b', b'new b contents\n'), ('tree2/c', b'new c contents\n'), ('tree2/d', b'c contents\n')])
    tree1.add(['b', 'c'], ids=[b'b1-id', b'c1-id'])
    tree2.add(['a', 'b', 'c', 'd'], ids=[b'b1-id', b'b2-id', b'c2-id', b'c1-id'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    expected = self.sorted([self.renamed(tree1, tree2, 'b', 'a', False), self.renamed(tree1, tree2, 'c', 'd', False), self.added(tree2, 'b'), self.added(tree2, 'c')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))