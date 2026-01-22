from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_starting_at_embedded_merge(self):
    branch = self.make_branch_with_embedded_merges()
    self.assertIterRevids(['4', '2.1.3', '2.2.1', '2.1.2', '2.1.1', '3', '2', '1.1.1', '1'], branch)
    self.assertIterRevids(['2.2.1', '2.1.1', '2', '1.1.1', '1'], branch, start_revision_id='2.2.1', stop_rule='with-merges')