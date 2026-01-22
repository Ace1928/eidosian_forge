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
def create_vhd(self, new_vhd_path, new_vhd_type, src_path=None, max_internal_size=0, parent_path=None, guid=None):
    new_device_id = self._get_vhd_device_id(new_vhd_path)
    vst = vdisk_struct.VIRTUAL_STORAGE_TYPE(DeviceId=new_device_id, VendorId=w_const.VIRTUAL_STORAGE_TYPE_VENDOR_MICROSOFT)
    params = vdisk_struct.CREATE_VIRTUAL_DISK_PARAMETERS()
    params.Version = w_const.CREATE_VIRTUAL_DISK_VERSION_2
    params.Version2.MaximumSize = max_internal_size
    params.Version2.ParentPath = parent_path
    params.Version2.SourcePath = src_path
    params.Version2.PhysicalSectorSizeInBytes = VIRTUAL_DISK_DEFAULT_PHYS_SECTOR_SIZE
    params.Version2.BlockSizeInBytes = w_const.CREATE_VHD_PARAMS_DEFAULT_BLOCK_SIZE
    params.Version2.SectorSizeInBytes = VIRTUAL_DISK_DEFAULT_SECTOR_SIZE
    if guid:
        params.Version2.UniqueId = wintypes.GUID.from_str(guid)
    handle = wintypes.HANDLE()
    create_virtual_disk_flag = CREATE_VIRTUAL_DISK_FLAGS.get(new_vhd_type, 0)
    self._run_and_check_output(virtdisk.CreateVirtualDisk, ctypes.byref(vst), ctypes.c_wchar_p(new_vhd_path), 0, None, create_virtual_disk_flag, 0, ctypes.byref(params), None, ctypes.byref(handle), cleanup_handle=handle)