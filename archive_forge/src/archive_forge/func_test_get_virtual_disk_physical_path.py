import ctypes
import os
from unittest import mock
import uuid
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.virtdisk import vhdutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(vhdutils.VHDUtils, '_open')
def test_get_virtual_disk_physical_path(self, mock_open):
    self._ctypes_patcher.stop()
    vhdutils.wintypes = wintypes
    fake_drive_path = '\\\\.\\PhysicialDrive5'

    def fake_run(func, handle, disk_path_sz_p, disk_path, **kwargs):
        disk_path_sz = ctypes.cast(disk_path_sz_p, wintypes.PULONG).contents.value
        self.assertEqual(w_const.MAX_PATH, disk_path_sz)
        disk_path.value = fake_drive_path
    self._mock_run.side_effect = fake_run
    ret_val = self._vhdutils.get_virtual_disk_physical_path(mock.sentinel.vhd_path)
    self.assertEqual(fake_drive_path, ret_val)
    mock_open.assert_called_once_with(mock.sentinel.vhd_path, open_flag=w_const.OPEN_VIRTUAL_DISK_FLAG_NO_PARENTS, open_access_mask=w_const.VIRTUAL_DISK_ACCESS_GET_INFO | w_const.VIRTUAL_DISK_ACCESS_DETACH)
    self._mock_run.assert_called_once_with(vhdutils.virtdisk.GetVirtualDiskPhysicalPath, mock_open.return_value, mock.ANY, mock.ANY, **self._run_args)