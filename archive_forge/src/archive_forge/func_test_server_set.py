import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_set(self):
    """Test server create, delete, set, show"""
    cmd_output = self.server_create()
    name = cmd_output['name']
    flavor = self.openstack('flavor show ' + self.flavor_name, parse_output=True)
    self.assertEqual(self.flavor_name, flavor['name'])
    self.assertEqual('%s (%s)' % (flavor['name'], flavor['id']), cmd_output['flavor'])
    image = self.openstack('image show ' + self.image_name, parse_output=True)
    self.assertEqual(self.image_name, image['name'])
    self.assertEqual('%s (%s)' % (image['name'], image['id']), cmd_output['image'])
    raw_output = self.openstack('server set ' + '--property a=b --property c=d ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('server show ' + name, parse_output=True)
    self.assertEqual({'a': 'b', 'c': 'd'}, cmd_output['properties'])
    raw_output = self.openstack('server unset ' + '--property a ' + name)
    cmd_output = self.openstack('server show ' + name, parse_output=True)
    self.assertEqual({'c': 'd'}, cmd_output['properties'])
    new_name = uuid.uuid4().hex
    raw_output = self.openstack('server set ' + '--name ' + new_name + ' ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('server show ' + new_name, parse_output=True)
    self.assertEqual(new_name, cmd_output['name'])
    raw_output = self.openstack('server set ' + '--name ' + name + ' ' + new_name)
    self.assertOutput('', raw_output)