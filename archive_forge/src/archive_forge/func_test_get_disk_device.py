from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
def test_get_disk_device(self):
    disk_device = mock.Mock()
    disk_device.__class__.__name__ = 'VirtualDisk'
    controller_device = mock.Mock()
    controller_device.__class__.__name__ = 'VirtualLSILogicController'
    devices = mock.Mock()
    devices.__class__.__name__ = 'ArrayOfVirtualDevice'
    devices.VirtualDevice = [disk_device, controller_device]
    session = mock.Mock()
    session.invoke_api.return_value = devices
    backing = mock.sentinel.backing
    self.assertEqual(disk_device, self._connector._get_disk_device(session, backing))
    session.invoke_api.assert_called_once_with(vim_util, 'get_object_property', session.vim, backing, 'config.hardware.device')