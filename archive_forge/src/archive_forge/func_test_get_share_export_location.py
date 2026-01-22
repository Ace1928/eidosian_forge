import ddt
from oslo_utils import uuidutils
from manilaclient.tests.functional import base
@ddt.data('admin', 'user')
def test_get_share_export_location(self, role):
    self.skip_if_microversion_not_supported('2.14')
    client = self.admin_client if role == 'admin' else self.user_client
    export_locations = client.list_share_export_locations(self.share['id'])
    el = client.get_share_export_location(self.share['id'], export_locations[0]['ID'])
    expected_keys = ['path', 'updated_at', 'created_at', 'id', 'preferred']
    if role == 'admin':
        expected_keys.extend(['is_admin_only', 'share_instance_id'])
    for key in expected_keys:
        self.assertIn(key, el)
    if role == 'admin':
        self.assertTrue(uuidutils.is_uuid_like(el['share_instance_id']))
        self.assertIn(el['is_admin_only'], ('True', 'False'))
    self.assertTrue(uuidutils.is_uuid_like(el['id']))
    self.assertIn(el['preferred'], ('True', 'False'))
    for list_k, get_k in (('ID', 'id'), ('Path', 'path'), ('Preferred', 'preferred')):
        self.assertEqual(export_locations[0][list_k], el[get_k])