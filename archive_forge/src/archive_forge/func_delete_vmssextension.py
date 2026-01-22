from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_vmssextension(self):
    self.log('Deleting vmextension {0}'.format(self.name))
    try:
        poller = self.compute_client.virtual_machine_scale_set_extensions.begin_delete(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name, vmss_extension_name=self.name)
        self.get_poller_result(poller)
    except Exception as e:
        self.log('Error attempting to delete the vmextension.')
        self.fail('Error deleting the vmextension: {0}'.format(str(e)))