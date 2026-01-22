from .. import mutabletree, tests
def test_with_pending_merges(self):
    self.tree.commit('first commit')
    other_tree = self.tree.controldir.sprout('other').open_workingtree()
    other_tree.commit('mergeable commit')
    self.tree.merge_from_branch(other_tree.branch)
    self.assertTrue(self.tree.has_changes())