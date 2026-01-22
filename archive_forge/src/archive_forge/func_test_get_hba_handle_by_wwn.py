import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_utils.FCUtils, '_open_adapter_by_wwn')
@mock.patch.object(fc_utils.FCUtils, '_close_adapter')
def test_get_hba_handle_by_wwn(self, mock_close_adapter, mock_open_adapter):
    with self._fc_utils._get_hba_handle(adapter_wwn_struct=mock.sentinel.wwn) as handle:
        self.assertEqual(mock_open_adapter.return_value, handle)
        mock_open_adapter.assert_called_once_with(mock.sentinel.wwn)
    mock_close_adapter.assert_called_once_with(mock_open_adapter.return_value)