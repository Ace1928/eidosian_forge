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
@mock.patch.object(executor.Executor, '_execute')
def test__get_fs_type(self, mock_execute):
    mock_execute.return_value = ('expected\n', '')
    result = self.connector._get_fs_type(NVME_DEVICE_PATH)
    self.assertEqual('expected', result)
    mock_execute.assert_called_once_with('blkid', NVME_DEVICE_PATH, '-s', 'TYPE', '-o', 'value', run_as_root=True, root_helper=self.connector._root_helper, check_exit_code=False)