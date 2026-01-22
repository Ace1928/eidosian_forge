import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def provision_vm(self, vm_name, fsk_filepath, pdk_filepath):
    vm = self._lookup_vm_check(vm_name)
    provisioning_service = self._conn_msps.Msps_ProvisioningService
    job_path, ret_val = provisioning_service.ProvisionMachine(fsk_filepath, vm.ConfigurationID, pdk_filepath)
    self._jobutils.check_ret_val(ret_val, job_path)