from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_aks(self):
    """
        Deletes the specified managed container service (AKS) in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the AKS instance {0}'.format(self.name))
    try:
        poller = self.managedcluster_client.managed_clusters.begin_delete(self.resource_group, self.name)
        self.get_poller_result(poller)
        return True
    except Exception as e:
        self.log('Error attempting to delete the AKS instance.')
        self.fail('Error deleting the AKS instance: {0}'.format(e.message))
        return False