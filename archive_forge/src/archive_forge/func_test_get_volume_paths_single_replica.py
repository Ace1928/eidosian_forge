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
@mock.patch.object(nvmeof.NVMeOFConnector, '_is_raid_device')
@mock.patch.object(nvmeof.NVMeOFConnProps, 'get_devices')
def test_get_volume_paths_single_replica(self, mock_get_devs, mock_is_raid):
    """Search for device from replicated conn props with 1 replica."""
    dev_path = '/dev/nvme1n1'
    mock_get_devs.return_value = [dev_path]
    target_props = volume_replicas[0]
    connection_properties = {'vol_uuid': VOL_UUID, 'alias': 'fakealias', 'volume_replicas': [target_props], 'replica_count': 1}
    conn_props = nvmeof.NVMeOFConnProps(connection_properties)
    res = self.connector.get_volume_paths(conn_props, None)
    self.assertEqual(['/dev/md/fakealias'], res)
    mock_is_raid.assert_called_once_with(dev_path)
    mock_get_devs.assert_called_once_with()