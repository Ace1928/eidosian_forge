import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_branches_no_working_trees(self):
    branch1_tree, branch2_tree = self.example_branches()
    dir1 = branch1_tree.controldir
    dir1.destroy_workingtree()
    self.assertFalse(dir1.has_workingtree())
    self.check_b2_vs_b1('diff --old branch2 --new branch1')
    self.check_b2_vs_b1('diff --old branch2 branch1')
    self.check_b2_vs_b1('diff branch2 --new branch1')
    self.check_b1_vs_b2('diff --old branch1 --new branch2')
    self.check_b1_vs_b2('diff --old branch1 branch2')
    self.check_b1_vs_b2('diff branch1 --new branch2')
    dir2 = branch2_tree.controldir
    dir2.destroy_workingtree()
    self.assertFalse(dir2.has_workingtree())
    self.check_b1_vs_b2('diff --old branch1 --new branch2')
    self.check_b1_vs_b2('diff --old branch1 branch2')
    self.check_b1_vs_b2('diff branch1 --new branch2')