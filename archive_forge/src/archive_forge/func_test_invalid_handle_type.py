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
def test_invalid_handle_type(self):
    self.assertRaises(exceptions.Invalid, self._cmgr._open(handle_type=None).__enter__)
    self.assertRaises(exceptions.Invalid, self._cmgr._close, mock.sentinel.handle, handle_type=None)