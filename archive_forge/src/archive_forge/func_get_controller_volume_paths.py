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
def get_controller_volume_paths(self, controller_path):
    disks = self._conn.query("SELECT * FROM %(class_name)s WHERE ResourceSubType = '%(res_sub_type)s' AND Parent='%(parent)s'" % {'class_name': self._RESOURCE_ALLOC_SETTING_DATA_CLASS, 'res_sub_type': self._PHYS_DISK_RES_SUB_TYPE, 'parent': controller_path})
    disk_data = {}
    for disk in disks:
        if disk.HostResource:
            disk_data[disk.path().RelPath] = disk.HostResource[0]
    return disk_data