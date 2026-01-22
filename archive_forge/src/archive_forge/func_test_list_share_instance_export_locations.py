import ddt
from oslo_utils import uuidutils
from manilaclient.tests.functional import base
def test_list_share_instance_export_locations(self):
    self.skip_if_microversion_not_supported('2.14')
    client = self.admin_client
    share_instances = client.list_share_instances(self.share['id'])
    self.assertGreater(len(share_instances), 0)
    self.assertIn('ID', share_instances[0])
    self.assertTrue(uuidutils.is_uuid_like(share_instances[0]['ID']))
    share_instance_id = share_instances[0]['ID']
    export_locations = client.list_share_instance_export_locations(share_instance_id)
    self.assertGreater(len(export_locations), 0)
    expected_keys = ('ID', 'Path', 'Is Admin only', 'Preferred')
    for el in export_locations:
        for key in expected_keys:
            self.assertIn(key, el)
        self.assertTrue(uuidutils.is_uuid_like(el['ID']))