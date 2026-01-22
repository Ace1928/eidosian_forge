import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_open_containing_through_symlink(self):
    self.make_test_tree()
    self.check_open_containing('link/content', 'tree', 'content')
    self.check_open_containing('link/sublink', 'tree', 'sublink')
    self.check_open_containing('link/sublink/subcontent', 'tree', 'sublink/subcontent')