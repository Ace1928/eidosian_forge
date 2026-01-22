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
def test_get_vhdx_current_header(self):
    fake_seq_numbers = [bytearray(b'\x01\x00\x00\x00\x00\x00\x00\x00'), bytearray(b'\x02\x00\x00\x00\x00\x00\x00\x00')]
    mock_handle = self._get_mock_file_handle(*fake_seq_numbers)
    offset = self._vhdutils._get_vhdx_current_header_offset(mock_handle)
    self.assertEqual(vhdutils.VHDX_HEADER_OFFSETS[1], offset)