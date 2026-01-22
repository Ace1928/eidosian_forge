import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_list_with_changes_before_and_changes_since(self):
    """Test server list.

        Getting the servers list with updated_at time equal or before than
        changes-before and equal or later than changes-since.
        """
    cmd_output = self.server_create()
    server_name1 = cmd_output['name']
    cmd_output = self.server_create()
    server_name2 = cmd_output['name']
    updated_at2 = cmd_output['updated']
    cmd_output = self.server_create()
    server_name3 = cmd_output['name']
    updated_at3 = cmd_output['updated']
    cmd_output = self.openstack('--os-compute-api-version 2.66 ' + 'server list ' + '--changes-since ' + updated_at2 + ' --changes-before ' + updated_at3, parse_output=True)
    col_updated = [server['Name'] for server in cmd_output]
    self.assertNotIn(server_name1, col_updated)
    self.assertIn(server_name2, col_updated)
    self.assertIn(server_name3, col_updated)