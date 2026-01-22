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
@mock.patch('glob.glob')
def test_get_all_namespaces_ctrl_paths(self, mock_glob):
    expected = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1', '/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2']
    mock_glob.return_value = expected[:]
    self.portal.controller = 'nvme0'
    res = self.portal.get_all_namespaces_ctrl_paths()
    self.assertEqual(expected, res)
    mock_glob.assert_called_once_with('/sys/class/nvme-fabrics/ctl/nvme0/nvme*')