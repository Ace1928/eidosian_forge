from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_NIC(self, vm_name, nic_name):
    return self.get_VM(vm_name).nics.get(nic_name)