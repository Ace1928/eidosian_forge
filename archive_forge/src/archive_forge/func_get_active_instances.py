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
def get_active_instances(self):
    """Return the names of all the active instances known to Hyper-V."""
    vm_names = self.list_instances()
    vms = [self._lookup_vm(vm_name, as_vssd=False) for vm_name in vm_names]
    active_vm_names = [v.ElementName for v in vms if v.EnabledState == constants.HYPERV_VM_STATE_ENABLED]
    return active_vm_names