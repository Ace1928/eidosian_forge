from breezy.revision import NULL_REVISION
from breezy.tests.per_repository import TestCaseWithRepository
def test_superset(self):
    tree = self.make_branch_and_tree('.')
    repo = tree.branch.repository
    rev1 = tree.commit('1')
    rev2 = tree.commit('2')
    rev3 = tree.commit('3')
    self.assertEqual({rev1, rev3}, repo.has_revisions([rev1, rev3, b'foobar:']))