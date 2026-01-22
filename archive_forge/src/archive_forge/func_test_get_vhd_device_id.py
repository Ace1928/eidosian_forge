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
@mock.patch.object(vhdutils.VHDUtils, 'get_vhd_format')
def test_get_vhd_device_id(self, mock_get_vhd_fmt):
    mock_get_vhd_fmt.return_value = constants.DISK_FORMAT_VHD
    dev_id = self._vhdutils._get_vhd_device_id(mock.sentinel.vhd_path)
    mock_get_vhd_fmt.assert_called_once_with(mock.sentinel.vhd_path)
    self.assertEqual(w_const.VIRTUAL_STORAGE_TYPE_DEVICE_VHD, dev_id)