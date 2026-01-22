from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def destroy_existing_planned_vm(self, vm_name):
    planned_vm = self._get_planned_vm(vm_name, self._compat_conn)
    if planned_vm:
        self._destroy_planned_vm(planned_vm)