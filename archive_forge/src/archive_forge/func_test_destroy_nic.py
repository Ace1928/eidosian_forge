from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_nic_data_by_name')
def test_destroy_nic(self, mock_get_nic_data_by_name):
    mock_nic_data = mock_get_nic_data_by_name.return_value
    self._vmutils._jobutils.remove_virt_resource.side_effect = exceptions.NotFound(message='fake_exc')
    self._vmutils.destroy_nic(self._FAKE_VM_NAME, mock.sentinel.FAKE_NIC_NAME)
    self._vmutils._jobutils.remove_virt_resource.assert_called_once_with(mock_nic_data)