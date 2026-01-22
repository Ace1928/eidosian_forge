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
@ddt.data(([], True), (['/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n2'], False))
@ddt.unpack
@mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
def test__can_disconnect_not_1_namespace(self, ctrl_paths, expected, mock_paths):
    """Check if can disconnect when we don't have 1 namespace in subsys."""
    self.portal.controller = 'nvme0'
    mock_paths.return_value = ctrl_paths
    res = self.portal.can_disconnect()
    self.assertIs(expected, res)
    mock_paths.assert_called_once_with()