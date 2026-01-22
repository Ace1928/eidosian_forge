import glob
import http.client
import queue
from unittest import mock
from unittest.mock import mock_open
from os_brick import exception
from os_brick.initiator.connectors import lightos
from os_brick.initiator import linuxscsi
from os_brick.privileged import lightos as priv_lightos
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(utils, 'get_host_nqn', return_value=FAKE_NQN)
@mock.patch.object(lightos.priv_lightos, 'move_dsc_file', return_value='/etc/discovery_client/discovery.d/v0')
@mock.patch.object(lightos.priv_lightos, 'delete_dsc_file', return_value=None)
@mock.patch.object(lightos.LightOSConnector, '_get_device_by_uuid', return_value=None)
def test_connect_volume_failure(self, mock_nqn, mock_move_file, mock_delete_file, mock_get_device):
    self.assertRaises(exception.BrickException, self.connector.connect_volume, self._get_connection_info())