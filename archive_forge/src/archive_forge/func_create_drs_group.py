from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_drs_group(self):
    """
        Function to create a DRS host/vm group
        """
    if self.__vm_list is None:
        self.__create_host_group()
    elif self.__host_list is None:
        self.__create_vm_group()
    else:
        raise Exception('Failed, no hosts or vms defined')