import uuid
from osc_placement.tests.functional import base
def test_fail_if_incorrect_aggregate_uuid(self):
    rp = self.resource_provider_create()
    aggs = ['abc', 'efg']
    self.assertCommandFailed("is not a 'uuid'", self.resource_provider_aggregate_set, rp['uuid'], *aggs, generation=rp['generation'])