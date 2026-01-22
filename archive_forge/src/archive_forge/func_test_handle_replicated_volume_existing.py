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
@mock.patch.object(nvmeof.NVMeOFConnector, 'stop_and_assemble_raid')
@mock.patch.object(nvmeof.NVMeOFConnector, '_is_device_in_raid')
def test_handle_replicated_volume_existing(self, mock_device_raid, mock_stop_assemble_raid):
    mock_device_raid.return_value = True
    conn_props = nvmeof.NVMeOFConnProps(connection_properties)
    result = self.connector._handle_replicated_volume(['/dev/nvme1n1', '/dev/nvme1n2', '/dev/nvme1n3'], conn_props)
    self.assertEqual('/dev/md/fakealias', result)
    mock_device_raid.assert_called_with('/dev/nvme1n1')
    mock_stop_assemble_raid.assert_called_with(['/dev/nvme1n1', '/dev/nvme1n2', '/dev/nvme1n3'], '/dev/md/fakealias', False)