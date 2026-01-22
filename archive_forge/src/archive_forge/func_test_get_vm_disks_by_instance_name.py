from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_vm_disks')
def test_get_vm_disks_by_instance_name(self, mock_get_vm_disks):
    self._lookup_vm()
    mock_get_vm_disks.return_value = mock.sentinel.vm_disks
    vm_disks = self._vmutils.get_vm_disks(self._FAKE_VM_NAME)
    self._vmutils._lookup_vm_check.assert_called_once_with(self._FAKE_VM_NAME)
    self.assertEqual(mock.sentinel.vm_disks, vm_disks)