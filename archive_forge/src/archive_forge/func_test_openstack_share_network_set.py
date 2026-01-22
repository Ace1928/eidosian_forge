from manilaclient.tests.functional.osc import base
def test_openstack_share_network_set(self):
    share_network = self.create_share_network()
    self.openstack('share network set %s --name %s' % (share_network['id'], 'new_name'))
    result = self.dict_result('share network', 'show %s' % share_network['id'])
    self.assertEqual(share_network['id'], result['id'])
    self.assertEqual('new_name', result['name'])