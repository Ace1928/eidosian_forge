import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
def test_share_access_list_with_filters(self):
    share = self.create_share()
    access_to_filter = '20.0.0.0/0'
    self.create_share_access_rule(share=share['name'], access_type='ip', access_to='0.0.0.0/0', wait=True)
    self.create_share_access_rule(share=share['name'], access_type='ip', access_to=access_to_filter, wait=True)
    output = self.openstack('share', params=f'access list {share['id']} --access-to {access_to_filter}', flags='--os-share-api-version 2.82')
    access_rule_list = self.parser.listing(output)
    self.assertTrue(len(access_rule_list) == 1)