import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
@_utils.not_found_decorator()
@_utils.retry_decorator(exceptions=exceptions.HyperVException)
def remove_multiple_virt_resources(self, virt_resources):
    job, ret_val = self._vs_man_svc.RemoveResourceSettings(ResourceSettings=[r.path_() for r in virt_resources])
    self.check_ret_val(ret_val, job)