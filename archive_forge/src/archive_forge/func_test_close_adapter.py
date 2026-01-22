import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
def test_close_adapter(self):
    self._fc_utils._close_adapter(mock.sentinel.hba_handle)
    fc_utils.hbaapi.HBA_CloseAdapter.assert_called_once_with(mock.sentinel.hba_handle)