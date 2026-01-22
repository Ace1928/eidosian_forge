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
@mock.patch.object(nvmeof.Target, 'set_portals_controllers', mock.Mock())
@mock.patch.object(nvmeof.NVMeOFConnector, '_try_disconnect_all')
@mock.patch.object(nvmeof.NVMeOFConnector, '_connect_target')
def test_connect_volume_not_replicated_fails(self, mock_connect_target, mock_disconnect):
    """Single vol attach fails and disconnects on failure."""
    connection_properties = volume_replicas[0].copy()
    mock_connect_target.side_effect = (exception.VolumeDeviceNotFound,)
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector.connect_volume, connection_properties)
    mock_connect_target.assert_called_with(mock.ANY)
    self.assertIsInstance(mock_connect_target.call_args[0][0], nvmeof.Target)
    mock_disconnect.assert_called_with(mock.ANY)
    self.assertIsInstance(mock_disconnect.call_args[0][0], nvmeof.NVMeOFConnProps)