from openstack import exceptions
from openstack.tests.functional import base
def test_create_network_provider_flat(self):
    existing_public = self.operator_cloud.search_networks(filters={'provider:network_type': 'flat'})
    if existing_public:
        self.skipTest('Physical network already allocated')
    net1 = self.operator_cloud.create_network(name=self.network_name, shared=True, provider={'physical_network': 'public', 'network_type': 'flat'})
    self.assertIn('id', net1)
    self.assertEqual(self.network_name, net1['name'])
    self.assertEqual('flat', net1['provider:network_type'])
    self.assertEqual('public', net1['provider:physical_network'])
    self.assertIsNone(net1['provider:segmentation_id'])