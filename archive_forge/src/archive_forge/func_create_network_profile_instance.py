from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_network_profile_instance(self, network):
    return self.managedcluster_models.ContainerServiceNetworkProfile(**network) if network else None