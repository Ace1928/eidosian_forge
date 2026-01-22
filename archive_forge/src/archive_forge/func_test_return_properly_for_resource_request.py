import operator
import uuid
from osc_placement.tests.functional import base
def test_return_properly_for_resource_request(self):
    rp1 = self.resource_provider_create()
    rp2 = self.resource_provider_create()
    self.resource_inventory_set(rp1['uuid'], 'PCI_DEVICE=8')
    self.resource_inventory_set(rp2['uuid'], 'PCI_DEVICE=16')
    rps = self.resource_provider_list(resources=['PCI_DEVICE=16'])
    self.assertEqual(1, len(rps))
    self.assertEqual(rp2['uuid'], rps[0]['uuid'])