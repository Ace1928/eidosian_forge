from __future__ import absolute_import, division, print_function
import copy
import datetime
import signal
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.teem import send_teem
def wait_for_device(self, start, end):
    while datetime.datetime.utcnow() < end:
        time.sleep(int(self.want.sleep))
        try:
            self.client = self._get_client_connection()
            if not self.client:
                continue
            if self._device_is_rebooting():
                continue
            if self.want.type == 'standard':
                if self._is_mprov_running_on_device():
                    self._wait_for_module_provisioning()
            elif self.want.type == 'vcmp':
                self._is_vcmpd_running_on_device()
            if not self._rest_endpoints_ready():
                self._wait_for_rest_interface()
            break
        except Exception as ex:
            if 'Failed to validate the SSL' in str(ex):
                raise F5ModuleError(str(ex))
            continue
    else:
        elapsed = datetime.datetime.utcnow() - start
        self.module.fail_json(msg=self.want.msg or 'Timeout when waiting for BIG-IP', elapsed=elapsed.seconds)