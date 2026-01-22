import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_create_with_empty_network_option_latest(self):
    """Test server create with empty network option in nova 2.latest."""
    server_name = uuid.uuid4().hex
    try:
        self.openstack('--os-compute-api-version 2.37 ' + 'server create ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + server_name)
    except exceptions.CommandFailed as e:
        self.assertNotIn('nics are required after microversion 2.36', e.stderr)