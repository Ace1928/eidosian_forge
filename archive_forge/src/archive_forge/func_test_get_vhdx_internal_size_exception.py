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
def test_get_vhdx_internal_size_exception(self):
    mock_open = self._mock_open()
    mock_open.side_effect = IOError
    func = self._vhdutils._get_internal_vhdx_size_by_file_size
    self.assertRaises(exceptions.VHDException, func, mock.sentinel.vhd_path, mock.sentinel.vhd_size, mock.sentinel.vhd_info)