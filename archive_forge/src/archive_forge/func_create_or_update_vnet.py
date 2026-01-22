from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, CIDR_PATTERN
def create_or_update_vnet(self, vnet):
    try:
        poller = self.network_client.virtual_networks.begin_create_or_update(self.resource_group, self.name, vnet)
        new_vnet = self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating virtual network {0} - {1}'.format(self.name, str(exc)))
    return virtual_network_to_dict(new_vnet)