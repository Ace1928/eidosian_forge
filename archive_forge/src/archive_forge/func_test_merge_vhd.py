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
@mock.patch('os.remove')
def test_merge_vhd(self, mock_remove, mock_open):
    open_params_struct = self._vdisk_struct.OPEN_VIRTUAL_DISK_PARAMETERS
    merge_params_struct = self._vdisk_struct.MERGE_VIRTUAL_DISK_PARAMETERS
    fake_open_params = open_params_struct.return_value
    fake_merge_params = merge_params_struct.return_value
    mock_open.return_value = mock.sentinel.handle
    self._vhdutils.merge_vhd(mock.sentinel.vhd_path)
    self.assertEqual(w_const.OPEN_VIRTUAL_DISK_VERSION_1, fake_open_params.Version)
    self.assertEqual(2, fake_open_params.Version1.RWDepth)
    mock_open.assert_called_once_with(mock.sentinel.vhd_path, open_params=self._ctypes.byref(fake_open_params))
    self.assertEqual(w_const.MERGE_VIRTUAL_DISK_VERSION_1, fake_merge_params.Version)
    self.assertEqual(1, fake_merge_params.Version1.MergeDepth)
    self._mock_run.assert_called_once_with(vhdutils.virtdisk.MergeVirtualDisk, mock.sentinel.handle, 0, self._ctypes.byref(fake_merge_params), None, **self._run_args)
    mock_remove.assert_called_once_with(mock.sentinel.vhd_path)
    self._mock_close.assert_called_once_with(mock.sentinel.handle)