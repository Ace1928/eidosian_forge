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
def test_get_iscsi_sessions(self):
    self._mock_ctypes()
    _get_iscsi_sessions = _utils.get_wrapped_function(self._initiator._get_iscsi_sessions)
    _get_iscsi_sessions(self._initiator, buff=mock.sentinel.buff, buff_size=mock.sentinel.buff_size, element_count=mock.sentinel.element_count)
    self._mock_run.assert_called_once_with(self._iscsidsc.GetIScsiSessionListW, self._ctypes.byref(mock.sentinel.buff_size), self._ctypes.byref(mock.sentinel.element_count), mock.sentinel.buff)