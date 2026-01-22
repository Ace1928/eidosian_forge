import uuid
from osc_placement.tests.functional import base
def test_member_of(self):
    rp1 = self.resource_provider_create()
    rp2 = self.resource_provider_create()
    self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
    self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
    agg1 = str(uuid.uuid4())
    agg2 = str(uuid.uuid4())
    agg3 = str(uuid.uuid4())
    self.resource_provider_aggregate_set(rp1['uuid'], agg1, agg3, generation=1)
    self.resource_provider_aggregate_set(rp2['uuid'], agg2, agg3, generation=1)
    agg1and3 = [agg1, agg3]
    agg1or3 = [agg1 + ',' + agg3]
    agg1or3_and_agg2 = [agg1 + ',' + agg3, agg2]
    rps = self.allocation_candidate_list(resources=('MEMORY_MB=1024',), member_of=agg1and3)
    candidate_dict = {rp['resource provider']: rp for rp in rps}
    self.assertEqual(1, len(candidate_dict))
    self.assertIn(rp1['uuid'], candidate_dict)
    rps = self.allocation_candidate_list(resources=('MEMORY_MB=1024',), member_of=agg1or3)
    candidate_dict = {rp['resource provider']: rp for rp in rps}
    self.assertEqual(2, len(candidate_dict))
    self.assertIn(rp1['uuid'], candidate_dict)
    self.assertIn(rp2['uuid'], candidate_dict)
    rps = self.allocation_candidate_list(resources=('MEMORY_MB=1024',), member_of=agg1or3_and_agg2)
    candidate_dict = {rp['resource provider']: rp for rp in rps}
    self.assertEqual(1, len(candidate_dict))
    self.assertIn(rp2['uuid'], candidate_dict)