from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_new_vm_info(self, vm):
    self.destination_content = connect_to_api(self.module, hostname=self.destination_vcenter, username=self.destination_vcenter_username, password=self.destination_vcenter_password, port=self.destination_vcenter_port, validate_certs=self.destination_vcenter_validate_certs)
    info = {}
    vm_obj = find_vm_by_name(content=self.destination_content, vm_name=vm)
    if vm_obj is None:
        self.module.fail_json(msg='Newly cloned VM is not found in the destination VCenter')
    else:
        vm_facts = gather_vm_facts(self.destination_content, vm_obj)
        info['vm_name'] = vm
        info['vcenter'] = self.destination_vcenter
        info['host'] = vm_facts['hw_esxi_host']
        info['datastore'] = vm_facts['hw_datastores']
        info['vm_folder'] = vm_facts['hw_folder']
        info['power_on'] = vm_facts['hw_power_status']
    return info