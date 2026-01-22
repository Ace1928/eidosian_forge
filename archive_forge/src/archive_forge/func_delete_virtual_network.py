from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, CIDR_PATTERN
def delete_virtual_network(self):
    try:
        poller = self.network_client.virtual_networks.begin_delete(self.resource_group, self.name)
        result = self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error deleting virtual network {0} - {1}'.format(self.name, str(exc)))
    return result