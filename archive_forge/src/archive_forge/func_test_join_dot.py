import os
from breezy import osutils, tests, workingtree
def test_join_dot(self):
    base_tree, sub_tree = self.make_trees()
    self.run_bzr('join .', working_dir='tree/subtree')
    self.check_success('tree')