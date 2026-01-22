from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data(True, False)
@mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
@mock.patch.object(vmutils.VMUtils, '_get_disk_controller_type')
@mock.patch.object(vmutils.VMUtils, '_get_wmi_obj')
@mock.patch.object(vmutils.VMUtils, '_get_disk_ctrl_addr')
def test_get_disk_attachment_info(self, is_physical, mock_get_disk_ctrl_addr, mock_get_wmi_obj, mock_get_disk_ctrl_type, mock_get_disk_res):
    mock_res = mock_get_disk_res.return_value
    exp_res = mock_res if is_physical else mock_get_wmi_obj.return_value
    fake_slot = 5
    exp_res.AddressOnParent = str(fake_slot)
    exp_att_info = dict(controller_slot=fake_slot, controller_path=exp_res.Parent, controller_type=mock_get_disk_ctrl_type.return_value, controller_addr=mock_get_disk_ctrl_addr.return_value)
    att_info = self._vmutils.get_disk_attachment_info(mock.sentinel.disk_path, is_physical)
    self.assertEqual(exp_att_info, att_info)
    if not is_physical:
        mock_get_wmi_obj.assert_called_once_with(mock_res.Parent)
    mock_get_disk_ctrl_type.assert_called_once_with(exp_res.Parent)
    mock_get_disk_ctrl_addr.assert_called_once_with(exp_res.Parent)