from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def planned_vm_exists(self, vm_name):
    """Checks if the Planned VM with the given name exists on the host."""
    return self._get_planned_vm(vm_name) is not None