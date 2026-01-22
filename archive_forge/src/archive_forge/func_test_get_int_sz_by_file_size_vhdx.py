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
def test_get_int_sz_by_file_size_vhdx(self):
    self._test_get_int_sz_by_file_size(vhd_dev_id=w_const.VIRTUAL_STORAGE_TYPE_DEVICE_VHDX)