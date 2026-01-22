from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_vnet_connection(self):
    connections = self.list_vnet_connections()
    for connection in connections:
        if connection.is_swift:
            return connection
    return None