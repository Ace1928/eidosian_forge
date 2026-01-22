import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def test_get_property_list(self):
    entry_0 = self._clusapi_utils.get_property_list_entry(name='fake prop name', syntax=1, value=ctypes.c_uint(2))
    entry_1 = self._clusapi_utils.get_property_list_entry(name='fake prop name', syntax=2, value=ctypes.c_ubyte(5))
    prop_list = self._clusapi_utils.get_property_list([entry_0, entry_1])
    self.assertEqual(2, prop_list.count)
    self.assertEqual(bytearray(entry_0) + bytearray(entry_1), prop_list.entries_buff)