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
def is_virtual_disk_file_attached(self, vhd_path):
    if not os.path.exists(vhd_path):
        LOG.debug('Image %s could not be found.', vhd_path)
        return False
    try:
        vhd_info = self.get_vhd_info(vhd_path, [w_const.GET_VIRTUAL_DISK_INFO_IS_LOADED])
        return bool(vhd_info['IsLoaded'])
    except Exception as exc:
        LOG.info('Could not get virtual disk information: %(vhd_path)s. Trying an alternative approach. Error: %(exc)s', dict(vhd_path=vhd_path, exc=exc))
        return self._disk_utils.is_virtual_disk_file_attached(vhd_path)