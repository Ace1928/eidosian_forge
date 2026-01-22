import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_merged_revtree_to_tree(self):
    tree = self.make_workingtree('a')
    tree.commit('first post')
    tree.commit('tree 1 commit 2')
    tree2 = self.make_workingtree('b')
    tree2.pull(tree.branch)
    tree2.commit('tree 2 commit 2')
    tree.merge_from_branch(tree2.branch)
    second_parent_tree = tree.revision_tree(tree.get_parent_ids()[1])
    second_parent_tree.lock_read()
    tree.lock_read()
    optimiser = InterTree.get(second_parent_tree, tree)
    tree.unlock()
    second_parent_tree.unlock()
    self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)