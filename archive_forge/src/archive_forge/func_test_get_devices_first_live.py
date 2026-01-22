import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(nvmeof.Target, 'present_portals', new_callable=mock.PropertyMock)
@mock.patch.object(nvmeof.Target, 'live_portals', new_callable=mock.PropertyMock)
def test_get_devices_first_live(self, mock_live, mock_present):
    """Return on first live portal with a device."""
    portal1 = mock.Mock(**{'get_device.return_value': None})
    portal2 = mock.Mock(**{'get_device.return_value': '/dev/nvme0n1'})
    portal3 = mock.Mock(**{'get_device.return_value': None})
    mock_live.return_value = [portal1, portal2]
    res = self.target.get_devices(only_live=True, get_one=True)
    self.assertListEqual(['/dev/nvme0n1'], res)
    mock_live.assert_called_once_with()
    mock_present.assert_not_called()
    portal1.get_device.assert_called_once_with()
    portal2.get_device.assert_called_once_with()
    portal3.get_device.assert_not_called()