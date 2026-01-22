from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def sanitize_params(self):
    """
        this method is used to verify user provided parameters
        """
    self.vm_obj = self.get_vm()
    if self.vm_obj is None:
        vm_id = self.vm_uuid or self.vm_name or self.moid
        self.module.fail_json(msg='Failed to find the VM/template with %s' % vm_id)
    self.destination_content = connect_to_api(self.module, hostname=self.destination_vcenter, username=self.destination_vcenter_username, password=self.destination_vcenter_password, port=self.destination_vcenter_port, validate_certs=self.destination_vcenter_validate_certs)
    vm = find_vm_by_name(content=self.destination_content, vm_name=self.params['destination_vm_name'])
    if vm:
        self.module.exit_json(changed=False, msg='A VM with the given name already exists')
    datastore_name = self.params['destination_datastore']
    datastore_cluster = find_obj(self.destination_content, [vim.StoragePod], datastore_name)
    if datastore_cluster:
        datastore_name = self.get_recommended_datastore(datastore_cluster_obj=datastore_cluster)
    self.destination_datastore = find_datastore_by_name(content=self.destination_content, datastore_name=datastore_name)
    if self.destination_datastore is None:
        self.module.fail_json(msg='Destination datastore not found.')
    self.destination_host = find_hostsystem_by_name(content=self.destination_content, hostname=self.params['destination_host'])
    if self.destination_host is None:
        self.module.fail_json(msg='Destination host not found.')
    if self.params['destination_resource_pool']:
        self.destination_resource_pool = find_resource_pool_by_name(content=self.destination_content, resource_pool_name=self.params['destination_resource_pool'])
    else:
        self.destination_resource_pool = self.destination_host.parent.resourcePool