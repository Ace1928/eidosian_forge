import ctypes
import os
import struct
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import virtdisk as vdisk_struct
from os_win.utils.winapi import wintypes
def get_internal_vhd_size_by_file_size(self, vhd_path, new_vhd_file_size):
    """Get internal size of a VHD according to new VHD file size."""
    vhd_info = self.get_vhd_info(vhd_path)
    vhd_type = vhd_info['ProviderSubtype']
    vhd_dev_id = vhd_info['DeviceId']
    if vhd_type == constants.VHD_TYPE_DIFFERENCING:
        vhd_parent = vhd_info['ParentPath']
        return self.get_internal_vhd_size_by_file_size(vhd_parent, new_vhd_file_size)
    if vhd_dev_id == w_const.VIRTUAL_STORAGE_TYPE_DEVICE_VHD:
        func = self._get_internal_vhd_size_by_file_size
    else:
        func = self._get_internal_vhdx_size_by_file_size
    return func(vhd_path, new_vhd_file_size, vhd_info)