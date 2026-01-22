from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def delete_natgateway(self):
    """
        Deletes specified NAT Gateway instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the NAT Gateway instance {0}'.format(self.name))
    try:
        response = self.network_client.nat_gateways.begin_delete(resource_group_name=self.resource_group, nat_gateway_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the NAT Gateway instance.')
        self.fail('Error deleting the NAT Gateway instance: {0}'.format(str(e)))
    return True