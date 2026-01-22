import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def create_two_trees_for_merging(self):
    """Create two trees that can be merged from.

        This sets self.tree_from, self.first_rev, self.tree_to, self.second_rev
        and self.to_second_rev.
        """
    self.tree_from = self.make_branch_and_tree('from')
    self.first_rev = self.tree_from.commit('first post')
    self.tree_to = self.tree_from.controldir.sprout('to').open_workingtree()
    self.second_rev = self.tree_from.commit('second rev on from', allow_pointless=True)
    self.to_second_rev = self.tree_to.commit('second rev on to', allow_pointless=True)