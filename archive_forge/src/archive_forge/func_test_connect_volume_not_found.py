from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import fibre_channel as fc
from os_brick.tests.windows import test_base
@mock.patch.object(fc.WindowsFCConnector, 'get_volume_paths')
def test_connect_volume_not_found(self, mock_get_vol_paths):
    mock_get_vol_paths.return_value = []
    self.assertRaises(exception.NoFibreChannelVolumeDeviceFound, self._connector.connect_volume, mock.sentinel.conn_props)