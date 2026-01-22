import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_snapshot_instance_export_locations_with_columns(self):
    client = self.admin_client
    snapshot_instances = client.list_snapshot_instances(self.snapshot['id'])
    self.assertGreater(len(snapshot_instances), 0)
    self.assertIn('ID', snapshot_instances[0])
    self.assertTrue(uuidutils.is_uuid_like(snapshot_instances[0]['ID']))
    snapshot_instance_id = snapshot_instances[0]['ID']
    export_locations = client.list_snapshot_instance_export_locations(snapshot_instance_id, columns='id,path')
    self.assertGreater(len(export_locations), 0)
    expected_keys = ('Id', 'Path')
    unexpected_keys = ('Updated At', 'Created At', 'Is Admin only')
    for el in export_locations:
        for key in expected_keys:
            self.assertIn(key, el)
        for key in unexpected_keys:
            self.assertNotIn(key, el)
        self.assertTrue(uuidutils.is_uuid_like(el['Id']))