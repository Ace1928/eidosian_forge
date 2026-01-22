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
def reconnect_parent_vhd(self, child_path, parent_path):
    open_params = vdisk_struct.OPEN_VIRTUAL_DISK_PARAMETERS()
    open_params.Version = w_const.OPEN_VIRTUAL_DISK_VERSION_2
    open_params.Version2.GetInfoOnly = False
    handle = self._open(child_path, open_flag=w_const.OPEN_VIRTUAL_DISK_FLAG_NO_PARENTS, open_access_mask=0, open_params=ctypes.byref(open_params))
    params = vdisk_struct.SET_VIRTUAL_DISK_INFO()
    params.Version = w_const.SET_VIRTUAL_DISK_INFO_PARENT_PATH
    params.ParentFilePath = parent_path
    self._run_and_check_output(virtdisk.SetVirtualDiskInformation, handle, ctypes.byref(params), cleanup_handle=handle)