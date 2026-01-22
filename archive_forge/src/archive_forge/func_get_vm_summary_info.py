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
@_utils.not_found_decorator(translated_exc=exceptions.HyperVVMNotFoundException)
def get_vm_summary_info(self, vm_name):
    vmsettings = self._lookup_vm_check(vm_name)
    settings_paths = [vmsettings.path_()]
    ret_val, summary_info = self._vs_man_svc.GetSummaryInformation([constants.VM_SUMMARY_NUM_PROCS, constants.VM_SUMMARY_ENABLED_STATE, constants.VM_SUMMARY_MEMORY_USAGE, constants.VM_SUMMARY_UPTIME], settings_paths)
    if ret_val:
        raise exceptions.HyperVException(_('Cannot get VM summary data for: %s') % vm_name)
    si = summary_info[0]
    memory_usage = None
    if si.MemoryUsage is not None:
        memory_usage = int(si.MemoryUsage)
    up_time = None
    if si.UpTime is not None:
        up_time = int(si.UpTime)
    enabled_state = self._enabled_states_map.get(si.EnabledState, constants.HYPERV_VM_STATE_ENABLED)
    summary_info_dict = {'NumberOfProcessors': si.NumberOfProcessors, 'EnabledState': enabled_state, 'MemoryUsage': memory_usage, 'UpTime': up_time}
    return summary_info_dict