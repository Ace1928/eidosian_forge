import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_dirstate_doesnt_read_from_repo_when_returning_cache_tree(self):
    """Getting parent trees from a dirstate tree does not read from the
        repos inventory store. This is an important part of the dirstate
        performance optimisation work.
        """
    tree = self.make_workingtree()
    subtree = self.make_branch_and_tree('subdir')
    subtree.lock_write()
    self.addCleanup(subtree.unlock)
    rev1 = subtree.commit('commit in subdir')
    rev1_tree = subtree.basis_tree()
    rev1_tree.lock_read()
    rev1_tree.root_inventory
    self.addCleanup(rev1_tree.unlock)
    rev2 = subtree.commit('second commit in subdir', allow_pointless=True)
    rev2_tree = subtree.basis_tree()
    rev2_tree.lock_read()
    rev2_tree.root_inventory
    self.addCleanup(rev2_tree.unlock)
    tree.branch.pull(subtree.branch)
    repo = tree.branch.repository
    self.overrideAttr(repo, 'get_inventory', self.fail)
    self.overrideAttr(repo, '_get_inventory_xml', self.fail)
    tree.set_parent_trees([(rev1, rev1_tree), (rev2, rev2_tree)])
    result_rev1_tree = tree.revision_tree(rev1)
    result_rev2_tree = tree.revision_tree(rev2)
    self.assertTreesEqual(rev2_tree, result_rev2_tree)
    self.assertRaises(errors.NoSuchRevisionInTree, self.assertTreesEqual, rev1_tree, result_rev1_tree)