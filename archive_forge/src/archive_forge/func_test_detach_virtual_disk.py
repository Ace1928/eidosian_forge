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
@ddt.data(True, False)
@mock.patch('os.path.exists')
@mock.patch.object(vhdutils.VHDUtils, '_open')
def test_detach_virtual_disk(self, exists, mock_open, mock_exists):
    mock_exists.return_value = exists
    self._mock_run.return_value = w_const.ERROR_NOT_READY
    self._vhdutils.detach_virtual_disk(mock.sentinel.vhd_path)
    mock_exists.assert_called_once_with(mock.sentinel.vhd_path)
    if exists:
        mock_open.assert_called_once_with(mock.sentinel.vhd_path, open_access_mask=w_const.VIRTUAL_DISK_ACCESS_DETACH)
        self._mock_run.assert_called_once_with(vhdutils.virtdisk.DetachVirtualDisk, mock_open.return_value, 0, 0, ignored_error_codes=[w_const.ERROR_NOT_READY], **self._run_args)
        self._mock_close.assert_called_once_with(mock_open.return_value)
    else:
        mock_open.assert_not_called()