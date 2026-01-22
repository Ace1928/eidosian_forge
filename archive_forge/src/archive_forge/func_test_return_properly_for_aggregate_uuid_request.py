import operator
import uuid
from osc_placement.tests.functional import base
def test_return_properly_for_aggregate_uuid_request(self):
    self.resource_provider_create()
    rp2 = self.resource_provider_create()
    agg = str(uuid.uuid4())
    self.resource_provider_aggregate_set(rp2['uuid'], agg)
    rps, warning = self.resource_provider_list(aggregate_uuids=[agg, str(uuid.uuid4())], may_print_to_stderr=True)
    self.assertEqual(1, len(rps))
    self.assertEqual(rp2['uuid'], rps[0]['uuid'])
    self.assertIn('The --aggregate-uuid option is deprecated, please use --member-of instead.', warning)
    rps = self.resource_provider_list(member_of=[agg])
    self.assertEqual(1, len(rps))
    self.assertEqual(rp2['uuid'], rps[0]['uuid'])