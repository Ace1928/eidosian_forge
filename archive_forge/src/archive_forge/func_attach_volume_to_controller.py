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
def attach_volume_to_controller(self, vm_name, controller_path, address, mounted_disk_path, serial=None):
    """Attach a volume to a controller."""
    vmsettings = self._lookup_vm_check(vm_name)
    diskdrive = self._get_new_resource_setting_data(self._PHYS_DISK_RES_SUB_TYPE)
    diskdrive.AddressOnParent = address
    diskdrive.Parent = controller_path
    diskdrive.HostResource = [mounted_disk_path]
    diskdrive_path = self._jobutils.add_virt_resource(diskdrive, vmsettings)[0]
    if serial:
        diskdrive = self._get_wmi_obj(diskdrive_path, True)
        diskdrive.ElementName = serial
        self._jobutils.modify_virt_resource(diskdrive)