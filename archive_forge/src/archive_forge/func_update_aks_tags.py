from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_aks_tags(self):
    try:
        poller = self.managedcluster_client.managed_clusters.begin_update_tags(self.resource_group, self.name, self.tags)
        response = self.get_poller_result(poller)
        return response.tags
    except Exception as exc:
        self.fail('Error attempting to update AKS tags: {0}'.format(exc.message))