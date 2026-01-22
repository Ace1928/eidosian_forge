import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_all_snapshot_instances(self):
    snapshot_instances = self.admin_client.list_snapshot_instances()
    self.assertGreater(len(snapshot_instances), 0)
    expected_keys = ('ID', 'Snapshot ID', 'Status')
    for si in snapshot_instances:
        for key in expected_keys:
            self.assertIn(key, si)
        self.assertTrue(uuidutils.is_uuid_like(si['ID']))
        self.assertTrue(uuidutils.is_uuid_like(si['Snapshot ID']))