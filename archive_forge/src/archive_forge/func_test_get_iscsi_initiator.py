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
def test_get_iscsi_initiator(self):
    self._mock_ctypes()
    self._ctypes.c_wchar = mock.MagicMock()
    fake_buff = (self._ctypes.c_wchar * (w_const.MAX_ISCSI_NAME_LEN + 1))()
    fake_buff.value = mock.sentinel.buff_value
    resulted_iscsi_initiator = self._initiator.get_iscsi_initiator()
    self._mock_run.assert_called_once_with(self._iscsidsc.GetIScsiInitiatorNodeNameW, fake_buff)
    self.assertEqual(mock.sentinel.buff_value, resulted_iscsi_initiator)