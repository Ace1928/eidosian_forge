from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def get_virtual_network(self, name):
    try:
        vnet = self.network_client.virtual_networks.get(self.virtual_network_resource_group, name)
        return vnet
    except ResourceNotFoundError as exc:
        self.fail('Error fetching virtual network {0} - {1}'.format(name, str(exc)))