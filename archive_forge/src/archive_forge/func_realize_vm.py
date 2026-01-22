from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def realize_vm(self, vm_name):
    planned_vm = self._get_planned_vm(vm_name, fail_if_not_found=True)
    if planned_vm:
        job_path, ret_val = self._vs_man_svc.ValidatePlannedSystem(planned_vm.path_())
        self._jobutils.check_ret_val(ret_val, job_path)
        job_path, ref, ret_val = self._vs_man_svc.RealizePlannedSystem(planned_vm.path_())
        self._jobutils.check_ret_val(ret_val, job_path)