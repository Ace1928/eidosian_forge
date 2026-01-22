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
def detach_vm_disk(self, vm_name, disk_path=None, is_physical=True, serial=None):
    disk_resource = self._get_mounted_disk_resource_from_path(disk_path, is_physical, serial=serial)
    if disk_resource:
        parent = self._conn.query("SELECT * FROM Msvm_ResourceAllocationSettingData WHERE __PATH = '%s'" % disk_resource.Parent)[0]
        self._jobutils.remove_virt_resource(disk_resource)
        if not is_physical:
            self._jobutils.remove_virt_resource(parent)