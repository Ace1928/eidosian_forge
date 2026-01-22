from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_create(self):
    share_name = 'test_create_share'
    share = self.create_share(name=share_name)
    self.assertEqual(share['share_proto'], 'NFS')
    self.assertEqual(share['size'], '1')
    self.assertEqual(share['name'], share_name)
    shares_list = self.listing_result('share', 'list')
    self.assertIn(share['id'], [item['ID'] for item in shares_list])