from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
from ansible.module_utils._text import to_native
def get_public_ip_address(self, name):
    self.log('Fetching public ip address {0}'.format(name))
    try:
        return self.network_client.public_ip_addresses.get(self.resource_group, name)
    except ResourceNotFoundError as exc:
        return None