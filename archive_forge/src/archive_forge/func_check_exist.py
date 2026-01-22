import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def check_exist(self, tree):
    """Just check that both files have the right executable bits set"""
    tree.lock_read()
    self.assertTrue(tree.is_executable('a'), "'a' lost the execute bit")
    self.assertFalse(tree.is_executable('b'), "'b' gained an execute bit")
    tree.unlock()