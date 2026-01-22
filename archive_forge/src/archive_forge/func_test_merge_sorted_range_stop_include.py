from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_range_stop_include(self):
    self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='2', stop_rule='include')