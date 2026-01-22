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
@mock.patch.object(nvmeof.Portal, 'get_device')
@mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
def test__can_disconnect(self, mock_paths, mock_device):
    """Can disconnect if the namespace is the one from this target.

        This tests that even when ANA is enabled it can identify the control
        path as belonging to the used device path.
        """
    self.portal.controller = 'nvme0'
    mock_device.return_value = '/dev/nvme1n2'
    mock_paths.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme1c1n2']
    self.assertTrue(self.portal.can_disconnect())