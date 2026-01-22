from manilaclient.tests.functional.osc import base
def test_openstack_share_network_unset(self):
    share_network = self.create_share_network(name='test_name')
    result1 = self.dict_result('share network', 'show %s' % share_network['id'])
    self.assertEqual(share_network['id'], result1['id'])
    self.assertEqual(share_network['name'], result1['name'])
    self.openstack('share network unset %s --name' % share_network['id'])
    result2 = self.dict_result('share network', 'show %s' % share_network['id'])
    self.assertEqual(share_network['id'], result2['id'])
    self.assertEqual('None', result2['name'])