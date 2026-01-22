import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data('admin', 'user')
def test_list_share_export_locations_with_columns(self, role):
    share, share_replica = self._create_share_and_replica()
    client = self.admin_client if role == 'admin' else self.user_client
    export_locations = client.list_share_replica_export_locations(share_replica['id'], columns='id,path')
    self.assertGreater(len(export_locations), 0)
    expected_keys = ('Id', 'Path')
    unexpected_keys = ('Updated At', 'Created At')
    for el in export_locations:
        for key in expected_keys:
            self.assertIn(key, el)
        for key in unexpected_keys:
            self.assertNotIn(key, el)
        self.assertTrue(uuidutils.is_uuid_like(el['Id']))