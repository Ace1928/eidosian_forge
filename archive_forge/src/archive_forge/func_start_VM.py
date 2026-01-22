from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def start_VM(self, vmname, timeout):
    VM = self.get_VM(vmname)
    try:
        VM.start()
    except Exception as e:
        setMsg('Failed to start VM.')
        setMsg(str(e))
        setFailed()
        return False
    return self.wait_VM(vmname, 'up', timeout)