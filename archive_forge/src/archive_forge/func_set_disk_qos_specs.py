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
def set_disk_qos_specs(self, disk_path, max_iops=None, min_iops=None):
    """Sets the disk's QoS policy."""
    if min_iops is None and max_iops is None:
        LOG.debug('Skipping setting disk QoS specs as no value was provided.')
        return
    disk_resource = self._get_mounted_disk_resource_from_path(disk_path, is_physical=False)
    if max_iops is not None:
        disk_resource.IOPSLimit = max_iops
    if min_iops is not None:
        disk_resource.IOPSReservation = min_iops
    self._jobutils.modify_virt_resource(disk_resource)