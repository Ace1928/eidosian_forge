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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, 'get_device_number_and_path')
def test_get_device_number_for_target(self, mock_get_dev_num_and_path):
    dev_num = self._initiator.get_device_number_for_target(mock.sentinel.target_name, mock.sentinel.lun, mock.sentinel.fail_if_not_found)
    mock_get_dev_num_and_path.assert_called_once_with(mock.sentinel.target_name, mock.sentinel.lun, mock.sentinel.fail_if_not_found)
    self.assertEqual(mock_get_dev_num_and_path.return_value[0], dev_num)