from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_api_server_access_profile_instance(self, server_access):
    return self.managedcluster_models.ManagedClusterAPIServerAccessProfile(**server_access) if server_access else None