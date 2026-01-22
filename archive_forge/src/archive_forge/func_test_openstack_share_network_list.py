from manilaclient.tests.functional.osc import base
def test_openstack_share_network_list(self):
    share_network = self.create_share_network()
    share_network_list = self.listing_result('share network', 'list')
    self.assertTableStruct(share_network_list, ['ID', 'Name'])
    self.assertIn(share_network['id'], [item['ID'] for item in share_network_list])