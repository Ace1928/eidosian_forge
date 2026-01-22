import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def get_disk_attachment_info(self, attached_disk_path=None, is_physical=True, serial=None):
    res = self._get_mounted_disk_resource_from_path(attached_disk_path, is_physical, serial=serial)
    if not res:
        err_msg = _("Disk '%s' is not attached to a vm.")
        raise exceptions.DiskNotFound(err_msg % attached_disk_path)
    if is_physical:
        drive = res
    else:
        drive = self._get_wmi_obj(res.Parent)
    ctrl_slot = int(drive.AddressOnParent)
    ctrl_path = drive.Parent
    ctrl_type = self._get_disk_controller_type(ctrl_path)
    ctrl_addr = self._get_disk_ctrl_addr(ctrl_path)
    attachment_info = dict(controller_slot=ctrl_slot, controller_path=ctrl_path, controller_type=ctrl_type, controller_addr=ctrl_addr)
    return attachment_info