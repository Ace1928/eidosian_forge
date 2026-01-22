import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('ID', 'Path')
def test_list_share_instances_by_export_location(self, option):
    export_locations = self.admin_client.list_share_export_locations(self.public_share['id'])
    share_instances = self.admin_client.list_share_instances(filters={'export_location': export_locations[0][option]})
    self.assertEqual(1, len(share_instances))
    share_instance_id = share_instances[0]['ID']
    except_export_locations = self.admin_client.list_share_instance_export_locations(share_instance_id)
    self.assertGreater(len(except_export_locations), 0)
    self.assertTrue(any((export_locations[0][option] == e[option] for e in except_export_locations)))