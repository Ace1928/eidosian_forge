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
def test__connect_volume_replicated_single_replica(self, mock_connect, mock_replicated, mock_single):
    """Connect to single repica backend."""
    conn_props = nvmeof.NVMeOFConnProps({'alias': 'fakealias', 'vol_uuid': VOL_UUID, 'volume_replicas': [volume_replicas[0]], 'replica_count': 1})
    found_devices = ['/dev/nvme0n1']
    mock_connect.side_effect = found_devices
    res = self.connector._connect_volume_replicated(conn_props)
    self.assertEqual(mock_single.return_value, res)
    mock_replicated.assert_not_called()
    mock_single.assert_called_once_with(found_devices, 'fakealias')