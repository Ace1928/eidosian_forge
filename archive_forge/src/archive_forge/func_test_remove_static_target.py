import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def test_remove_static_target(self):
    self._mock_ctypes()
    self._initiator._remove_static_target(mock.sentinel.target_name)
    expected_ignored_err_codes = [w_const.ISDSC_TARGET_NOT_FOUND]
    self._mock_run.assert_called_once_with(self._iscsidsc.RemoveIScsiStaticTargetW, self._ctypes.c_wchar_p(mock.sentinel.target_name), ignored_error_codes=expected_ignored_err_codes)