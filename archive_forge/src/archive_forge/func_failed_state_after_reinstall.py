from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def failed_state_after_reinstall(self, host, count=0):
    if host.status in [hoststate.ERROR, hoststate.INSTALL_FAILED, hoststate.NON_OPERATIONAL]:
        return self.raise_host_exception()
    if host.status == hoststate.NON_RESPONSIVE:
        if count <= 3:
            time.sleep(20)
            return self.failed_state_after_reinstall(self._service.service(host.id).get(), count + 1)
        else:
            return self.raise_host_exception()
    return False