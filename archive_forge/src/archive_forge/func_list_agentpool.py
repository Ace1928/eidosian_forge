from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_agentpool(self):
    result = []
    try:
        resp = self.managedcluster_client.agent_pools.list(self.resource_group, self.cluster_name)
        while True:
            result.append(resp.next())
    except StopIteration:
        pass
    except Exception:
        pass
    return result