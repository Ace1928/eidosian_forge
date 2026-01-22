from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def check_sync(self, exisiting_vnet_peering):
    if exisiting_vnet_peering['peering_sync_level'] == 'LocalNotInSync':
        return True
    return False