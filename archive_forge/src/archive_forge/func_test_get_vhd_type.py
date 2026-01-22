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
@mock.patch.object(vhdutils.VHDUtils, 'get_vhd_info')
def test_get_vhd_type(self, mock_get_vhd_info):
    mock_get_vhd_info.return_value = {'ProviderSubtype': mock.sentinel.provider_subtype}
    ret_val = self._vhdutils.get_vhd_type(mock.sentinel.vhd_path)
    self.assertEqual(mock.sentinel.provider_subtype, ret_val)
    mock_get_vhd_info.assert_called_once_with(mock.sentinel.vhd_path, [w_const.GET_VIRTUAL_DISK_INFO_PROVIDER_SUBTYPE])