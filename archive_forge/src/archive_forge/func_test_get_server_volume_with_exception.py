from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_get_server_volume_with_exception(self):
    self.assertRaises(TypeError, self.cs.volumes.get_server_volume, '1234')
    self.assertRaises(TypeError, self.cs.volumes.get_server_volume, '1234', volume_id='Work', attachment_id='123')