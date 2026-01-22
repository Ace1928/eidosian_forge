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
@mock.patch.object(nvmeof.Target, 'get_devices')
def test_find_device_not_found(self, mock_get_devs):
    """Finding a devices tries up to 5 times before giving up."""
    mock_get_devs.return_value = []
    self.assertRaises(exception.VolumeDeviceNotFound, self.target.find_device)
    self.assertEqual(5, mock_get_devs.call_count)
    mock_get_devs.assert_has_calls(5 * [mock.call(only_live=True, get_one=True)])