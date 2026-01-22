import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_utils.FCUtils, 'get_fc_hba_count')
def test_get_fc_hba_ports_missing_hbas(self, mock_get_fc_hba_count):
    mock_get_fc_hba_count.return_value = 0
    resulted_hba_ports = self._fc_utils.get_fc_hba_ports()
    self.assertEqual([], resulted_hba_ports)