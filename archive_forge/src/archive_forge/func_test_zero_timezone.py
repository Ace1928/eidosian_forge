from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_zero_timezone(self):
    tree1 = self.make_branch_and_tree('br1')
    r1 = tree1.commit(message='quux', timezone=0)
    rev_a = tree1.branch.repository.get_revision(r1)
    self.assertEqual(0, rev_a.timezone)