import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def check_open_containing(self, to_open, expected_tree_name, expected_relpath):
    wt, relpath = workingtree.WorkingTree.open_containing(to_open)
    self.assertEqual(relpath, expected_relpath)
    self.assertEndsWith(wt.basedir, expected_tree_name)