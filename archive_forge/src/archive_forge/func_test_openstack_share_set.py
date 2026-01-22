from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_set(self):
    share = self.create_share()
    self.openstack(f'share set {share['id']} --name new_name --property key=value')
    result = self.dict_result('share', f'show {share['id']}')
    self.assertEqual(share['id'], result['id'])
    self.assertEqual('new_name', result['name'])
    self.assertEqual("key='value'", result['properties'])