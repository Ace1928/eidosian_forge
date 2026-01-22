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
def test_get_md_name(self):
    mock_open = mock.mock_open(read_data=md_stat_contents)
    with mock.patch('builtins.open', mock_open):
        result = self.connector.get_md_name(os.path.basename(NVME_NS_PATH))
    self.assertEqual('md0', result)
    mock_open.assert_called_once_with('/proc/mdstat', 'r')
    mock_fd = mock_open.return_value.__enter__.return_value
    mock_fd.__iter__.assert_called_once_with()