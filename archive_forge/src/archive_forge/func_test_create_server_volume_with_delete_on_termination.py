from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_create_server_volume_with_delete_on_termination(self):
    v = self.cs.volumes.create_server_volume(server_id=1234, volume_id='15e59938-07d5-11e1-90e3-e3dffe0c5983', device='/dev/vdb', tag='tag1', delete_on_termination=True)
    self.assert_request_id(v, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('POST', '/servers/1234/os-volume_attachments', {'volumeAttachment': {'volumeId': '15e59938-07d5-11e1-90e3-e3dffe0c5983', 'device': '/dev/vdb', 'tag': 'tag1', 'delete_on_termination': True}})
    self.assertIsInstance(v, volumes.Volume)