from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.windows import base as base_win_conn
from os_brick.tests.windows import fake_win_conn
from os_brick.tests.windows import test_base
@mock.patch.object(fake_win_conn.FakeWindowsConnector, 'get_volume_paths')
def test_extend_volume_missing_path(self, mock_get_vol_paths):
    mock_get_vol_paths.return_value = []
    self.assertRaises(exception.NotFound, self._connector.extend_volume, mock.sentinel.conn_props)
    mock_get_vol_paths.assert_called_once_with(mock.sentinel.conn_props)