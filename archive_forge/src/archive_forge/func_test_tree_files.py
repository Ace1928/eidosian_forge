import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_tree_files(self):
    self.make_test_tree()
    self.check_tree_files(['tree/outerlink'], 'tree', ['outerlink'])
    self.check_tree_files(['link/outerlink'], 'tree', ['outerlink'])
    self.check_tree_files(['link/sublink/subcontent'], 'tree', ['subdir/subcontent'])