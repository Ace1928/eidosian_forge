from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
@mock.patch('warnings.warn')
def test_delete_server_volume_with_warn(self, mock_warn):
    self.cs.volumes.delete_server_volume(1234, volume_id=None, attachment_id='Work')
    mock_warn.assert_called_once()