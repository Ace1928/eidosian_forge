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
def test_guid_from_str(self):
    buff = list(range(16))
    py_uuid = uuid.UUID(bytes=bytes(buff))
    guid = wintypes.GUID.from_str(str(py_uuid))
    guid_bytes = ctypes.cast(ctypes.byref(guid), ctypes.POINTER(wintypes.BYTE * 16)).contents
    self.assertEqual(buff, guid_bytes[:])