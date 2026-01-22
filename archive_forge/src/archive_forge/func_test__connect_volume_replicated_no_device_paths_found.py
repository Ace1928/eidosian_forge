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
@mock.patch.object(nvmeof.NVMeOFConnector, '_handle_single_replica')
@mock.patch.object(nvmeof.NVMeOFConnector, '_handle_replicated_volume')
@mock.patch.object(nvmeof.NVMeOFConnector, '_connect_target')
def test__connect_volume_replicated_no_device_paths_found(self, mock_connect, mock_replicated, mock_single):
    """Fail if cannot connect to any replica."""
    mock_connect.side_effect = 3 * [Exception]
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector._connect_volume_replicated, CONN_PROPS)
    mock_replicated.assert_not_called()
    mock_single.assert_not_called()