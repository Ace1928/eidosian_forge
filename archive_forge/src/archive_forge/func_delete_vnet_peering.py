from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def delete_vnet_peering(self):
    """
        Deletes the specified Azure Virtual Network Peering

        :return: True
        """
    self.log('Deleting Azure Virtual Network Peering {0}'.format(self.name))
    try:
        poller = self.network_client.virtual_network_peerings.begin_delete(self.resource_group, self.virtual_network['name'], self.name)
        self.get_poller_result(poller)
        return True
    except Exception as e:
        self.fail('Error deleting the Azure Virtual Network Peering: {0}'.format(e.message))
        return False