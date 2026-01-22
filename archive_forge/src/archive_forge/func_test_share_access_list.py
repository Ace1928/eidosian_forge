import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
@ddt.data('2.45', '2.33', '2.21')
def test_share_access_list(self, microversion):
    share = self.create_share()
    self.create_share_access_rule(share=share['name'], access_type='ip', access_to='0.0.0.0/0', wait=True)
    output = self.openstack('share', params=f'access list {share['id']}', flags=f'--os-share-api-version {microversion}')
    access_rule_list = self.parser.listing(output)
    base_list = ['ID', 'Access Type', 'Access To', 'Access Level', 'State']
    if microversion >= '2.33':
        base_list.append('Access Key')
    if microversion >= '2.45':
        base_list.extend(['Created At', 'Updated At'])
    self.assertTableStruct(access_rule_list, base_list)
    self.assertTrue(len(access_rule_list) > 0)
    self.create_share_access_rule(share=share['name'], access_type='ip', access_to='192.168.0.151', wait=True, properties='foo=bar')
    output = self.openstack('share', params=f'access list {share['id']} --properties foo=bar', flags=f'--os-share-api-version {microversion}')
    access_rule_properties = self.parser.listing(output)
    self.assertEqual(1, len(access_rule_properties))
    self.assertEqual(access_rule_properties['id'], access_rule_properties[0]['ID'])