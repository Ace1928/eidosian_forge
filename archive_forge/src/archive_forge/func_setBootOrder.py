from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setBootOrder(self, vmname, boot_order):
    self.__get_conn()
    VM = self.conn.get_VM(vmname)
    bootorder = []
    for boot_dev in VM.os.get_boot():
        bootorder.append(str(boot_dev.dev))
    if boot_order != bootorder:
        self.conn.set_BootOrder(vmname, boot_order)
        setMsg('The boot order has been set')
    else:
        setMsg('The boot order has already been set')
    return True