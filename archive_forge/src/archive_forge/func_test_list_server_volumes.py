from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_list_server_volumes(self):
    vl = self.cs.volumes.get_server_volumes(1234)
    self.assert_request_id(vl, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/servers/1234/os-volume_attachments')
    for v in vl:
        self.assertIsInstance(v, volumes.Volume)