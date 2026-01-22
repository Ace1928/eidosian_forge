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
def test_get_vhd_format_by_sig_zero_length_file(self):
    mock_open = self._mock_open(read_data=('', ''))
    fmt = self._vhdutils._get_vhd_format_by_signature(mock.sentinel.vhd_path)
    self.assertIsNone(fmt)
    mock_open.return_value.seek.assert_called_once_with(0, 2)