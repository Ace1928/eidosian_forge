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
@mock.patch.object(vhdutils.VHDUtils, '_get_vhdx_current_header_offset')
def test_get_log_size(self, mock_get_vhdx_curr_hd_offset):
    fake_curr_header_offset = vhdutils.VHDX_HEADER_OFFSETS[0]
    fake_log_sz = bytearray(b'\x01\x00\x00\x00')
    mock_get_vhdx_curr_hd_offset.return_value = fake_curr_header_offset
    mock_handle = self._get_mock_file_handle(fake_log_sz)
    log_size = self._vhdutils._get_vhdx_log_size(mock_handle)
    self.assertEqual(log_size, 1)