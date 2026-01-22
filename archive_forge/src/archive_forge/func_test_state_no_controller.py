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
@mock.patch.object(nvmeof, 'ctrl_property')
def test_state_no_controller(self, mock_property):
    """Cannot read the state if the controller name has not been found."""
    self.portal.controller = None
    self.assertIsNone(self.portal.state)
    mock_property.assert_not_called()