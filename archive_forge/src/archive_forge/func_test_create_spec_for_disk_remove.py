from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
def test_create_spec_for_disk_remove(self):
    disk_spec = mock.Mock()
    session = mock.Mock()
    session.vim.client.factory.create.return_value = disk_spec
    disk_device = mock.sentinel.disk_device
    self._connector._create_spec_for_disk_remove(session, disk_device)
    session.vim.client.factory.create.assert_called_once_with('ns0:VirtualDeviceConfigSpec')
    self.assertEqual('remove', disk_spec.operation)
    self.assertEqual('destroy', disk_spec.fileOperation)
    self.assertEqual(disk_device, disk_spec.device)