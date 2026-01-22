from unittest import mock
import ddt
from oslo_concurrency import processutils
from os_brick import exception
from os_brick.initiator.windows import rbd
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.windows import test_base
@ddt.data(True, False)
@mock.patch.object(rbd.WindowsRBDConnector, 'get_device_name')
@mock.patch('oslo_utils.eventletutils.EventletEvent.wait')
def test_wait_for_volume(self, device_found, mock_wait, mock_get_dev_name):
    mock_open = mock.mock_open()
    if device_found:
        mock_get_dev_name.return_value = mock.sentinel.dev_name
    else:
        mock_get_dev_name.side_effect = [None] + [mock.sentinel.dev_name] * self._conn.device_scan_attempts
        mock_open.side_effect = FileNotFoundError
    with mock.patch.object(rbd, 'open', mock_open, create=True):
        if device_found:
            dev_name = self._conn._wait_for_volume(self.connection_properties)
            self.assertEqual(mock.sentinel.dev_name, dev_name)
        else:
            self.assertRaises(exception.VolumeDeviceNotFound, self._conn._wait_for_volume, self.connection_properties)
        mock_open.assert_any_call(mock.sentinel.dev_name, 'rb')
        mock_get_dev_name.assert_any_call(self.connection_properties, expect=False)