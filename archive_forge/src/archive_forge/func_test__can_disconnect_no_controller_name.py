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
@mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
def test__can_disconnect_no_controller_name(self, mock_paths):
    """Cannot disconnect when portal doesn't have a controller."""
    res = self.portal.can_disconnect()
    self.assertFalse(res)
    mock_paths.assert_not_called()