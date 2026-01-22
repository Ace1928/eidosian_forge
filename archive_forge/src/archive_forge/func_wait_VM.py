from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def wait_VM(self, vmname, state, timeout):
    VM = self.get_VM(vmname)
    while VM.status.state != state:
        VM = self.get_VM(vmname)
        time.sleep(10)
        if timeout is not False:
            timeout -= 10
            if timeout <= 0:
                setMsg('Timeout expired')
                setFailed()
                return False
    return True