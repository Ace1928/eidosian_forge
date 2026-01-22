from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def get_vnet_peering(self):
    """
        Gets the Virtual Network Peering.

        :return: deserialized Virtual Network Peering
        """
    self.log('Checking if Virtual Network Peering {0} is present'.format(self.name))
    try:
        response = self.network_client.virtual_network_peerings.get(self.resource_group, self.virtual_network['name'], self.name)
        self.log('Response : {0}'.format(response))
        return vnetpeering_to_dict(response)
    except ResourceNotFoundError:
        self.log('Did not find the Virtual Network Peering.')
        return False