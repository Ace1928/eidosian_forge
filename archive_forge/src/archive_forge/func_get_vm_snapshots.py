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
def get_vm_snapshots(self, vm_name, snapshot_name=None):
    vm = self._lookup_vm_check(vm_name, as_vssd=False)
    snapshots = vm.associators(wmi_association_class=self._VIRTUAL_SYSTEM_SNAP_ASSOC_CLASS, wmi_result_class=self._VIRTUAL_SYSTEM_SETTING_DATA_CLASS)
    return [s.path_() for s in snapshots if snapshot_name is None or s.ElementName == snapshot_name]