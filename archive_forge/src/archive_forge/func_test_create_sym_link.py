import ctypes
import os
import shutil
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import pathutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def test_create_sym_link(self):
    tg_is_dir = False
    self._pathutils.create_sym_link(mock.sentinel.path, mock.sentinel.target, target_is_dir=tg_is_dir)
    self._mock_run.assert_called_once_with(pathutils.kernel32.CreateSymbolicLinkW, mock.sentinel.path, mock.sentinel.target, tg_is_dir, kernel32_lib_func=True)