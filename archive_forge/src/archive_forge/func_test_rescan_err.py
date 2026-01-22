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
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
def test_rescan_err(self, mock_run_nvme_cli):
    """Test failure on nvme rescan subprocess execution."""
    mock_run_nvme_cli.side_effect = Exception()
    self.assertRaises(exception.CommandExecutionFailed, self.connector.rescan, 'nvme1')
    nvme_command = ('ns-rescan', NVME_DEVICE_PATH)
    mock_run_nvme_cli.assert_called_with(nvme_command)