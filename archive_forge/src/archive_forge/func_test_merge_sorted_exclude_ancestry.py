from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_exclude_ancestry(self):
    branch = self.make_branch_with_alternate_ancestries()
    self.assertIterRevids(['3', '1.1.2', '1.2.1', '2', '1.1.1', '1'], branch)
    self.assertIterRevids(['1.1.2', '1.2.1'], branch, stop_rule='with-merges-without-common-ancestry', start_revision_id='1.1.2', stop_revision_id='1.1.1')