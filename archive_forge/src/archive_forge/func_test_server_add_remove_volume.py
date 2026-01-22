import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_add_remove_volume(self):
    volume_wait_for = volume_common.BaseVolumeTests.wait_for_status
    server_name = uuid.uuid4().hex
    cmd_output = self.openstack('server create ' + '--network private ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + '--wait ' + server_name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(server_name, cmd_output['name'])
    self.addCleanup(self.openstack, 'server delete --wait ' + server_name)
    server_id = cmd_output['id']
    volume_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 1 ' + volume_name, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(volume_name, cmd_output['name'])
    volume_wait_for('volume', volume_name, 'available')
    self.addCleanup(self.openstack, 'volume delete ' + volume_name)
    volume_id = cmd_output['id']
    cmd_output = self.openstack('server add volume ' + server_name + ' ' + volume_name + ' ' + '--tag bar', parse_output=True)
    self.assertEqual(server_id, cmd_output['Server ID'])
    self.assertEqual(volume_id, cmd_output['Volume ID'])
    cmd_output = self.openstack('server volume list ' + server_name, parse_output=True)
    self.assertEqual(server_id, cmd_output[0]['Server ID'])
    self.assertEqual(volume_id, cmd_output[0]['Volume ID'])
    volume_wait_for('volume', volume_name, 'in-use')
    cmd_output = self.openstack('server event list ' + server_name, parse_output=True)
    self.assertEqual(2, len(cmd_output))
    self.assertIn('attach_volume', {x['Action'] for x in cmd_output})
    self.openstack('server remove volume ' + server_name + ' ' + volume_name)
    volume_wait_for('volume', volume_name, 'available')
    cmd_output = self.openstack('server event list ' + server_name, parse_output=True)
    self.assertEqual(3, len(cmd_output))
    self.assertIn('detach_volume', {x['Action'] for x in cmd_output})
    raw_output = self.openstack('server volume list ' + server_name)
    self.assertEqual('\n', raw_output)