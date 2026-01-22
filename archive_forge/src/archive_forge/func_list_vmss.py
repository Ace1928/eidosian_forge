from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
def list_vmss(self):
    self.log('List all')
    try:
        response = self.network_client.network_interfaces.list_virtual_machine_scale_set_network_interfaces(resource_group_name=self.resource_group, virtual_machine_scale_set_name=self.vmss_name)
        return list(response)
    except Exception as exc:
        self.fail('Error listing all - {0}'.format(str(exc)))