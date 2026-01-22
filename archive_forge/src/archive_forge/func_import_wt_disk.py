from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import hostutils
from os_win.utils import pathutils
from os_win.utils import win32utils
def import_wt_disk(self, vhd_path, wtd_name):
    """Import a vhd/x image to be used by Windows iSCSI targets."""
    try:
        self._conn_wmi.WT_Disk.ImportWTDisk(DevicePath=vhd_path, Description=wtd_name)
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Failed to import WT disk: %s.')
        raise exceptions.ISCSITargetWMIException(err_msg % vhd_path, wmi_exc=wmi_exc)