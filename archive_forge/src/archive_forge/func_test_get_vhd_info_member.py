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
@mock.patch.object(vhdutils.VHDUtils, '_parse_vhd_info')
def test_get_vhd_info_member(self, mock_parse_vhd_info):
    get_vd_info_struct = self._vdisk_struct.GET_VIRTUAL_DISK_INFO
    fake_params = get_vd_info_struct.return_value
    fake_info_size = self._ctypes.sizeof.return_value
    info_member = w_const.GET_VIRTUAL_DISK_INFO_PARENT_LOCATION
    vhd_info = self._vhdutils._get_vhd_info_member(mock.sentinel.vhd_path, info_member)
    self._mock_run.assert_called_once_with(vhdutils.virtdisk.GetVirtualDiskInformation, mock.sentinel.vhd_path, self._ctypes.byref(self._ctypes.c_ulong(fake_info_size)), self._ctypes.byref(fake_params), None, ignored_error_codes=[w_const.ERROR_VHD_INVALID_TYPE], **self._run_args)
    self.assertEqual(mock_parse_vhd_info.return_value, vhd_info)
    mock_parse_vhd_info.assert_called_once_with(fake_params, info_member)