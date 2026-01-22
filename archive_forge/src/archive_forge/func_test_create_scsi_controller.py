from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_new_resource_setting_data')
def test_create_scsi_controller(self, mock_get_new_rsd):
    mock_vm = self._lookup_vm()
    self._vmutils.create_scsi_controller(self._FAKE_VM_NAME)
    self._vmutils._jobutils.add_virt_resource.assert_called_once_with(mock_get_new_rsd.return_value, mock_vm)