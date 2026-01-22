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
def test_get_vhdx_metadata_size(self):
    fake_md_offset = bytearray(b'\x01\x00\x00\x00\x00\x00\x00\x00')
    fake_md_sz = bytearray(b'\x01\x00\x00\x00')
    mock_handle = self._get_mock_file_handle(fake_md_offset, fake_md_sz)
    md_sz, md_offset = self._vhdutils._get_vhdx_metadata_size_and_offset(mock_handle)
    self.assertEqual(1, md_sz)
    self.assertEqual(1, md_offset)