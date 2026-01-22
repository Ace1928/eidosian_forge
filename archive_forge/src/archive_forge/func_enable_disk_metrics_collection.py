from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def enable_disk_metrics_collection(self, attached_disk_path=None, is_physical=False, serial=None):
    disk = self._vmutils._get_mounted_disk_resource_from_path(attached_disk_path, is_physical=is_physical, serial=serial)
    self._enable_metrics(disk)