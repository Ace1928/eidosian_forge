import uuid
from osc_placement.tests.functional import base
def test_list_shared(self):
    rp1 = self.resource_provider_create()
    rp2 = self.resource_provider_create()
    self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192')
    self.resource_inventory_set(rp2['uuid'], 'DISK_GB=1024')
    agg = str(uuid.uuid4())
    self.resource_provider_aggregate_set(rp1['uuid'], agg)
    self.resource_provider_aggregate_set(rp2['uuid'], agg)
    self.resource_provider_trait_set(rp2['uuid'], 'MISC_SHARES_VIA_AGGREGATE')
    candidates = self.allocation_candidate_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'))
    rps = {c['resource provider']: c for c in candidates}
    self.assertResourceEqual('MEMORY_MB=1024', rps[rp1['uuid']]['allocation'])
    self.assertResourceEqual('DISK_GB=80', rps[rp2['uuid']]['allocation'])
    self.assertResourceEqual('MEMORY_MB=0/8192', rps[rp1['uuid']]['inventory used/capacity'])
    self.assertResourceEqual('DISK_GB=0/1024', rps[rp2['uuid']]['inventory used/capacity'])
    self.assertEqual(rps[rp2['uuid']]['#'], rps[rp1['uuid']]['#'])