from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data(constants.DISK, constants.DVD)
@mock.patch.object(vmutils.VMUtils, '_get_new_resource_setting_data')
def test_attach_drive(self, drive_type, mock_get_new_rsd):
    mock_vm = self._lookup_vm()
    mock_drive_res = mock.Mock()
    mock_disk_res = mock.Mock()
    mock_get_new_rsd.side_effect = [mock_drive_res, mock_disk_res]
    self._jobutils.add_virt_resource.side_effect = [[mock.sentinel.drive_res_path], [mock.sentinel.disk_res_path]]
    self._vmutils.attach_drive(mock.sentinel.vm_name, mock.sentinel.disk_path, mock.sentinel.ctrl_path, mock.sentinel.drive_addr, drive_type)
    self._vmutils._lookup_vm_check.assert_called_once_with(mock.sentinel.vm_name, as_vssd=False)
    if drive_type == constants.DISK:
        exp_res_sub_types = [self._vmutils._DISK_DRIVE_RES_SUB_TYPE, self._vmutils._HARD_DISK_RES_SUB_TYPE]
    else:
        exp_res_sub_types = [self._vmutils._DVD_DRIVE_RES_SUB_TYPE, self._vmutils._DVD_DISK_RES_SUB_TYPE]
    mock_get_new_rsd.assert_has_calls([mock.call(exp_res_sub_types[0]), mock.call(exp_res_sub_types[1], self._vmutils._STORAGE_ALLOC_SETTING_DATA_CLASS)])
    self.assertEqual(mock.sentinel.ctrl_path, mock_drive_res.Parent)
    self.assertEqual(mock.sentinel.drive_addr, mock_drive_res.Address)
    self.assertEqual(mock.sentinel.drive_addr, mock_drive_res.AddressOnParent)
    self.assertEqual(mock.sentinel.drive_res_path, mock_disk_res.Parent)
    self.assertEqual([mock.sentinel.disk_path], mock_disk_res.HostResource)
    self._jobutils.add_virt_resource.assert_has_calls([mock.call(mock_drive_res, mock_vm), mock.call(mock_disk_res, mock_vm)])