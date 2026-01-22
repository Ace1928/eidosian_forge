from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def find_vm(self, vmid):
    """
        Extra bonus feature: vmid = -1 returns a list of everything
        """
    vms = self.conn.listAllDomains()
    if vmid == -1:
        return vms
    for vm in vms:
        if vm.name() == vmid:
            return vm
    raise VMNotFound('virtual machine %s not found' % vmid)