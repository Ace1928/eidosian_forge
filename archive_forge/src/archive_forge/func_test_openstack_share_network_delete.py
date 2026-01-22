from manilaclient.tests.functional.osc import base
def test_openstack_share_network_delete(self):
    share_network = self.create_share_network(add_cleanup=False)
    share_network_list = self.listing_result('share network', 'list')
    self.assertIn(share_network['id'], [item['ID'] for item in share_network_list])
    self.openstack('share network delete %s' % share_network['id'])
    self.check_object_deleted('share network', share_network['id'])
    share_network_list_after_delete = self.listing_result('share network', 'list')
    self.assertNotIn(share_network['id'], [item['ID'] for item in share_network_list_after_delete])