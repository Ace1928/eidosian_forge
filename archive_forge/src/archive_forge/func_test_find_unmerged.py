from .. import missing, tests
from ..missing import iter_log_revisions
from . import TestCaseWithTransport
def test_find_unmerged(self):
    original_tree = self.make_branch_and_tree('original')
    original = original_tree.branch
    puller_tree = self.make_branch_and_tree('puller')
    puller = puller_tree.branch
    merger_tree = self.make_branch_and_tree('merger')
    merger = merger_tree.branch
    self.assertUnmerged(([], []), original, puller)
    original_tree.commit('a', rev_id=b'a')
    self.assertUnmerged(([('1', b'a', 0)], []), original, puller)
    puller_tree.pull(original)
    self.assertUnmerged(([], []), original, puller)
    merger_tree.pull(original)
    original_tree.commit('b', rev_id=b'b')
    original_tree.commit('c', rev_id=b'c')
    self.assertUnmerged(([('2', b'b', 0), ('3', b'c', 0)], []), original, puller)
    self.assertUnmerged(([('3', b'c', 0), ('2', b'b', 0)], []), original, puller, backward=True)
    puller_tree.pull(original)
    self.assertUnmerged(([], []), original, puller)
    self.assertUnmerged(([('2', b'b', 0), ('3', b'c', 0)], []), original, merger)
    merger_tree.merge_from_branch(original)
    self.assertUnmerged(([('2', b'b', 0), ('3', b'c', 0)], []), original, merger)
    merger_tree.commit('d', rev_id=b'd')
    self.assertUnmerged(([], [('2', b'd', 0)]), original, merger)