import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_get_history(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('1', rev_id=b'1', allow_pointless=True)
    tree.commit('2', rev_id=b'2', allow_pointless=True)
    tree.commit('3', rev_id=b'3', allow_pointless=True)
    rev = tree.branch.repository.get_revision(b'1')
    history = rev.get_history(tree.branch.repository)
    self.assertEqual([None, b'1'], history)
    rev = tree.branch.repository.get_revision(b'2')
    history = rev.get_history(tree.branch.repository)
    self.assertEqual([None, b'1', b'2'], history)
    rev = tree.branch.repository.get_revision(b'3')
    history = rev.get_history(tree.branch.repository)
    self.assertEqual([None, b'1', b'2', b'3'], history)