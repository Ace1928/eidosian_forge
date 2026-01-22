import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def make_branches(self, format=None):
    """Create two branches

    branch 1 has 6 commits, branch 2 has 3 commits
    commit 10 is a ghosted merge merge from branch 1

    the object graph is
    B:     A:
    a..0   a..0
    a..1   a..1
    a..2   a..2
    b..3   a..3 merges b..4
    b..4   a..4
    b..5   a..5 merges b..5
    b..6 merges a4

    so A is missing b6 at the start
    and B is missing a3, a4, a5
    """
    tree1 = self.make_branch_and_tree('branch1', format=format)
    br1 = tree1.branch
    tree1.commit('Commit one', rev_id=b'a@u-0-0')
    tree1.commit('Commit two', rev_id=b'a@u-0-1')
    tree1.commit('Commit three', rev_id=b'a@u-0-2')
    tree2 = tree1.controldir.sprout('branch2').open_workingtree()
    br2 = tree2.branch
    tree2.commit('Commit four', rev_id=b'b@u-0-3')
    tree2.commit('Commit five', rev_id=b'b@u-0-4')
    self.assertEqual(br2.last_revision(), b'b@u-0-4')
    tree1.merge_from_branch(br2)
    tree1.commit('Commit six', rev_id=b'a@u-0-3')
    tree1.commit('Commit seven', rev_id=b'a@u-0-4')
    tree2.commit('Commit eight', rev_id=b'b@u-0-5')
    self.assertEqual(br2.last_revision(), b'b@u-0-5')
    tree1.merge_from_branch(br2)
    tree1.commit('Commit nine', rev_id=b'a@u-0-5')
    br1.lock_read()
    try:
        graph = br1.repository.get_graph()
        revhistory = list(graph.iter_lefthand_ancestry(br1.last_revision(), [revision.NULL_REVISION]))
        revhistory.reverse()
    finally:
        br1.unlock()
    tree2.add_parent_tree_id(revhistory[4])
    tree2.commit('Commit ten - ghost merge', rev_id=b'b@u-0-6')
    return (br1, br2)