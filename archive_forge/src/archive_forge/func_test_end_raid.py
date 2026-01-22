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
@mock.patch.object(os.path, 'exists')
@mock.patch.object(nvmeof.NVMeOFConnector, 'stop_raid')
@mock.patch.object(nvmeof.NVMeOFConnector, 'is_raid_exists')
def test_end_raid(self, mock_raid_exists, mock_stop_raid, mock_os):
    mock_raid_exists.return_value = True
    mock_stop_raid.return_value = False
    mock_os.return_value = True
    self.assertIsNone(self.connector.end_raid('/dev/md/md1'))
    mock_raid_exists.assert_called_with('/dev/md/md1')
    mock_stop_raid.assert_called_with('/dev/md/md1', True)
    mock_os.assert_called_with('/dev/md/md1')