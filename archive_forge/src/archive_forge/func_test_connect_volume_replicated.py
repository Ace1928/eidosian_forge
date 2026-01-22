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
@mock.patch.object(nvmeof.NVMeOFConnector, '_connect_volume_replicated')
@mock.patch.object(nvmeof.NVMeOFConnector, '_connect_target')
def test_connect_volume_replicated(self, mock_connect_target, mock_replicated_volume, mock_disconnect):
    mock_replicated_volume.return_value = '/dev/md/md1'
    actual = self.connector.connect_volume(connection_properties)
    expected = {'type': 'block', 'path': '/dev/md/md1'}
    self.assertEqual(expected, actual)
    mock_replicated_volume.assert_called_once_with(mock.ANY)
    self.assertIsInstance(mock_replicated_volume.call_args[0][0], nvmeof.NVMeOFConnProps)
    mock_connect_target.assert_not_called()
    mock_disconnect.assert_not_called()