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
def test_parse_vhd_info(self):
    fake_info_member = w_const.GET_VIRTUAL_DISK_INFO_SIZE
    fake_info = mock.Mock()
    fake_info.Size._fields_ = [('VirtualSize', vhdutils.wintypes.ULARGE_INTEGER), ('PhysicalSize', vhdutils.wintypes.ULARGE_INTEGER)]
    fake_info.Size.VirtualSize = mock.sentinel.virt_size
    fake_info.Size.PhysicalSize = mock.sentinel.phys_size
    ret_val = self._vhdutils._parse_vhd_info(fake_info, fake_info_member)
    expected = {'VirtualSize': mock.sentinel.virt_size, 'PhysicalSize': mock.sentinel.phys_size}
    self.assertEqual(expected, ret_val)