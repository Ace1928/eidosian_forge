from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_vnet_connection(self):
    try:
        return self.web_client.web_apps.delete_swift_virtual_network(resource_group_name=self.resource_group, name=self.name)
    except Exception as exc:
        self.fail('Error deleting webapp vnet connection {0} (rg={1}) - {3}'.format(self.name, self.resource_group, str(exc)))