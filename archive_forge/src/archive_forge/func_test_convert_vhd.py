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
@mock.patch.object(vhdutils.VHDUtils, 'create_vhd')
def test_convert_vhd(self, mock_create_vhd):
    self._vhdutils.convert_vhd(mock.sentinel.src, mock.sentinel.dest, mock.sentinel.vhd_type)
    mock_create_vhd.assert_called_once_with(mock.sentinel.dest, mock.sentinel.vhd_type, src_path=mock.sentinel.src)