import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_show(self):
    """Test server show"""
    cmd_output = self.server_create()
    name = cmd_output['name']
    cmd_output = json.loads(self.openstack(f'server show -f json {name}'))
    self.assertEqual(name, cmd_output['name'])
    cmd_output = json.loads(self.openstack(f'server show -f json {name} --diagnostics'))
    self.assertIn('driver', cmd_output)
    cmd_output = json.loads(self.openstack(f'server show -f json {name} --topology --os-compute-api-version 2.78'))
    self.assertIn('topology', cmd_output)