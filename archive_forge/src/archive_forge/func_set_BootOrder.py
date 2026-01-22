from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_BootOrder(self, vmname, boot_order):
    VM = self.get_VM(vmname)
    bootorder = []
    for device in boot_order:
        bootorder.append(params.Boot(dev=device))
    VM.os.boot = bootorder
    try:
        VM.update()
        setChanged()
    except Exception as e:
        setMsg('Failed to update the boot order.')
        setMsg(str(e))
        setFailed()
        return False
    return True