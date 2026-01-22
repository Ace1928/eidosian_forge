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
@mock.patch.object(nvmeof, 'ctrl_property', return_value='10')
def test_reconnect_delay(self, mock_property):
    """Reconnect delay returns an int."""
    self.portal.controller = 'nvme0'
    self.assertIs(10, self.portal.reconnect_delay)
    mock_property.assert_called_once_with('reconnect_delay', 'nvme0')