from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_Memory(self, name, memory):
    VM = self.get_VM(name)
    VM.memory = int(int(memory) * 1024 * 1024 * 1024)
    try:
        VM.update()
        setMsg('The Memory has been updated.')
        setChanged()
        return True
    except Exception as e:
        setMsg('Failed to update memory.')
        setMsg(str(e))
        setFailed()
        return False