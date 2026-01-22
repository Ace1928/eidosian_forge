from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def get_public_ip_address_instance(self, id):
    """Get a reference to the public ip address resource"""
    self.log('Fetching public ip address {0}'.format(id))
    resource_id = format_resource_id(id, self.subscription_id, 'Microsoft.Network', 'publicIPAddresses', self.resource_group)
    return self.network_models.PublicIPAddress(id=resource_id)