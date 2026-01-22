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
@mock.patch.object(nvmeof.NVMeOFConnector, '_get_fs_type')
def test__is_raid_device(self, mock_get_fs_type):
    mock_get_fs_type.return_value = 'linux_raid_member'
    result = self.connector._is_raid_device(NVME_DEVICE_PATH)
    self.assertTrue(result)
    mock_get_fs_type.assert_called_once_with(NVME_DEVICE_PATH)