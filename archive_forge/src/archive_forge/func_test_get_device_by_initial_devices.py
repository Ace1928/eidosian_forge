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
@mock.patch.object(nvmeof.Target, 'get_device_path_by_initial_devices')
@mock.patch.object(nvmeof.Portal, 'get_device_by_property')
def test_get_device_by_initial_devices(self, mock_property, mock_get_dev):
    """With no id, calls target to get device from initial devices."""
    mock_get_dev.return_value = 'result'
    self.target.uuid = None
    self.target.nguid = None
    self.target.ns_id = None
    res = self.portal.get_device()
    self.assertEqual('result', res)
    mock_get_dev.assert_called_once_with()