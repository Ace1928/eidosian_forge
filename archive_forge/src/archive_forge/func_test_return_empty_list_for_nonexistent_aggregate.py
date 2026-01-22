import operator
import uuid
from osc_placement.tests.functional import base
def test_return_empty_list_for_nonexistent_aggregate(self):
    self.resource_provider_create()
    agg = str(uuid.uuid4())
    rps, warning = self.resource_provider_list(aggregate_uuids=[agg], may_print_to_stderr=True)
    self.assertEqual([], rps)
    self.assertIn('The --aggregate-uuid option is deprecated, please use --member-of instead.', warning)
    rps = self.resource_provider_list(member_of=[agg])
    self.assertEqual([], rps)