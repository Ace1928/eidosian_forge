from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def remove_CD(self, vmname):
    VM = self.get_VM(vmname)
    try:
        VM.cdroms.get(id='00000000-0000-0000-0000-000000000000').delete()
        setMsg('Removed the image.')
        setChanged()
    except Exception as e:
        setMsg('Failed to remove the image.')
        setMsg(str(e))
        setFailed()
        return False
    return True