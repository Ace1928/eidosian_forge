import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_unique_root_id_per_tree(self):
    format_name = 'development-subtree'
    tree1 = self.make_branch_and_tree('tree1', format=format_name)
    tree2 = self.make_branch_and_tree('tree2', format=format_name)
    self.assertNotEqual(tree1.path2id(''), tree2.path2id(''))
    tree1.commit('first post')
    tree3 = tree1.controldir.sprout('tree3').open_workingtree()
    self.assertEqual(tree3.path2id(''), tree1.path2id(''))