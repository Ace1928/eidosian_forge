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
@mock.patch.object(builtins, 'open')
def test_sysfs_property(self, mock_open):
    """Method is basically an open an read method."""
    mock_read = mock_open.return_value.__enter__.return_value.read
    mock_read.return_value = ' uuid '
    res = nvmeof.sysfs_property('uuid', '/sys/class/block/nvme0n1')
    self.assertEqual('uuid', res)
    mock_open.assert_called_once_with('/sys/class/block/nvme0n1/uuid', 'r')
    mock_read.assert_called_once_with()