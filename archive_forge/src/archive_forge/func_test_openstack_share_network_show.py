from manilaclient.tests.functional.osc import base
def test_openstack_share_network_show(self):
    share_network = self.create_share_network()
    result = self.dict_result('share network', 'show %s' % share_network['id'])
    self.assertEqual(share_network['id'], result['id'])
    listing_result = self.listing_result('share network', 'show %s' % share_network['id'])
    self.assertTableStruct(listing_result, ['Field', 'Value'])