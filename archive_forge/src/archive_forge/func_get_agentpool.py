from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_agentpool(self):
    try:
        return self.managedcluster_client.agent_pools.get(self.resource_group, self.cluster_name, self.name)
    except ResourceNotFoundError:
        pass