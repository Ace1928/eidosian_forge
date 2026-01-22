from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_update_server_volume(self):
    vol_id = '15e59938-07d5-11e1-90e3-e3dffe0c5983'
    v = self.cs.volumes.update_server_volume(server_id=1234, src_volid='Work', dest_volid=vol_id)
    self.assert_request_id(v, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('PUT', '/servers/1234/os-volume_attachments/Work')
    self.assertIsInstance(v, volumes.Volume)