import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_boot_from_volume(self):
    server_name = uuid.uuid4().hex
    server = self.openstack('server create ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + '--boot-from-volume 1 ' + self.network_arg + ' ' + '--wait ' + server_name, parse_output=True)
    self.assertIsNotNone(server['id'])
    self.assertEqual(server_name, server['name'])
    self.wait_for_status(server_name, 'ACTIVE')
    cmd_output = self.openstack('server show ' + server_name, parse_output=True)
    volumes_attached = cmd_output['volumes_attached']
    self.assertIsNotNone(volumes_attached)
    attached_volume_id = volumes_attached[0]['id']
    for vol in volumes_attached:
        self.assertIsNotNone(vol['id'])
        self.addCleanup(self.openstack, 'volume delete ' + vol['id'])
    self.assertEqual(v2_server.IMAGE_STRING_FOR_BFV, cmd_output['image'])
    cmd_output = self.openstack('volume show ' + volumes_attached[0]['id'], parse_output=True)
    self.assertEqual(1, int(cmd_output['size']))
    attachments = cmd_output['attachments']
    self.assertEqual(1, len(attachments))
    self.assertEqual(server['id'], attachments[0]['server_id'])
    self.assertEqual('in-use', cmd_output['status'])
    self.openstack('server delete --wait ' + server_name)
    cmd_output = self.openstack('volume show ' + attached_volume_id, parse_output=True)
    self.assertEqual('available', cmd_output['status'])